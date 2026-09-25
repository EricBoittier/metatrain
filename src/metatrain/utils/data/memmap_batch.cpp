// Batched loading for MemmapDataset: gathers the rows of a whole batch from
// the memory-mapped arrays in one pass, and builds the systems and the joined
// target TensorMaps directly, instead of one TensorMap per structure followed
// by metatensor.join. See metatrain/utils/data/memmap_batch.py for the Python
// side, which also holds a pure-Python equivalent of this file.

#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>

using metatensor_torch::Labels;
using metatensor_torch::LabelsHolder;
using metatensor_torch::TensorBlock;
using metatensor_torch::TensorBlockHolder;
using metatensor_torch::TensorMap;
using metatensor_torch::TensorMapHolder;
using metatomic_torch::System;
using metatomic_torch::SystemHolder;

namespace {

/// Where each structure of the batch lives in the per-atom arrays.
struct AtomRanges {
    std::vector<int64_t> start;
    std::vector<int64_t> count;
    int64_t total = 0;
};

AtomRanges atom_ranges(const torch::Tensor& na, const std::vector<int64_t>& indices) {
    auto cumulative = na.data_ptr<int64_t>();
    auto n_structures = na.size(0) - 1;
    auto ranges = AtomRanges();
    ranges.start.reserve(indices.size());
    ranges.count.reserve(indices.size());
    for (auto i: indices) {
        if (i < 0 || i >= n_structures) {
            C10_THROW_ERROR(IndexError,
                "structure index " + std::to_string(i) + " is out of range for a dataset of "
                + std::to_string(n_structures) + " structures"
            );
        }
        ranges.start.push_back(cumulative[i]);
        ranges.count.push_back(cumulative[i + 1] - cumulative[i]);
        ranges.total += ranges.count.back();
    }
    return ranges;
}

/// Number of scalars in one row (everything but the first dimension).
int64_t row_size(const torch::Tensor& array) {
    int64_t size = 1;
    for (int64_t d = 1; d < array.dim(); d++) {
        size *= array.size(d);
    }
    return size;
}

std::vector<int64_t> with_rows(const torch::Tensor& array, int64_t rows) {
    auto shape = array.sizes().vec();
    shape[0] = rows;
    return shape;
}

/// Copy `count[k]` rows starting at `start[k]` for every k, back to back,
/// converting to `Out`. `scale` multiplies every value.
template <typename In, typename Out>
void gather_rows(
    const torch::Tensor& array,
    const std::vector<int64_t>& start,
    const std::vector<int64_t>& count,
    Out* output,
    Out scale
) {
    auto input = array.data_ptr<In>();
    auto width = row_size(array);
    for (size_t k = 0; k < start.size(); k++) {
        auto source = input + start[k] * width;
        auto n = count[k] * width;
        for (int64_t e = 0; e < n; e++) {
            output[e] = scale * static_cast<Out>(source[e]);
        }
        output += n;
    }
}

/// Rows `start[k] .. start[k] + count[k]` of a float32 array, as float64.
torch::Tensor gather_float(
    const torch::Tensor& array,
    const std::vector<int64_t>& start,
    const std::vector<int64_t>& count,
    int64_t rows,
    double scale = 1.0
) {
    TORCH_CHECK(array.scalar_type() == torch::kFloat32 && array.is_contiguous(),
        "memmap arrays must be contiguous float32 tensors");
    auto output = torch::empty(with_rows(array, rows), torch::TensorOptions().dtype(torch::kFloat64));
    gather_rows<float, double>(array, start, count, output.data_ptr<double>(), scale);
    return output;
}

std::vector<int64_t> ones(size_t n) {
    return std::vector<int64_t>(n, 1);
}

Labels make_labels(std::vector<std::string> names, torch::Tensor values) {
    // every sample label built here is unique by construction: the structure
    // indices of a batch are checked for duplicates on the Python side
    return torch::make_intrusive<LabelsHolder>(
        torch::IValue(std::move(names)), std::move(values), metatensor::assume_unique{}
    );
}

/// `[[indices[k], j] for k, j in the atoms of structure k]`, as labels.
Labels per_atom_samples(
    std::vector<std::string> names,
    const std::vector<int64_t>& first,
    const AtomRanges& ranges
) {
    auto values = torch::empty({ranges.total, 2}, torch::TensorOptions().dtype(torch::kInt32));
    auto data = values.data_ptr<int32_t>();
    for (size_t k = 0; k < first.size(); k++) {
        for (int64_t j = 0; j < ranges.count[k]; j++) {
            *data++ = static_cast<int32_t>(first[k]);
            *data++ = static_cast<int32_t>(j);
        }
    }
    return make_labels(std::move(names), values);
}

Labels per_structure_samples(std::string name, const std::vector<int64_t>& values) {
    auto tensor = torch::empty({static_cast<int64_t>(values.size()), 1}, torch::TensorOptions().dtype(torch::kInt32));
    auto data = tensor.data_ptr<int32_t>();
    for (size_t k = 0; k < values.size(); k++) {
        data[k] = static_cast<int32_t>(values[k]);
    }
    return make_labels({std::move(name)}, tensor);
}

/// The component labels of an array shaped (samples, 3, ..., 3, properties).
std::vector<Labels> cartesian_components(int64_t dim) {
    auto rank = dim - 2;
    if (rank == 1) {
        return {LabelsHolder::range("xyz", 3)};
    }
    auto components = std::vector<Labels>();
    for (int64_t d = 0; d < rank; d++) {
        components.push_back(LabelsHolder::range("xyz_" + std::to_string(d + 1), 3));
    }
    return components;
}

} // namespace

/// Load structures `indices` of a MemmapDataset as systems, and each of the
/// `fields` (targets and extra data) as a single TensorMap for the batch.
///
/// The result matches collating the per-structure samples of
/// `MemmapDataset.__getitem__` with `group_and_join`: sample labels keep the
/// dataset indices, gradient samples count from 0 within the batch.
std::tuple<std::vector<System>, std::vector<TensorMap>> load_batch(
    torch::Tensor na,
    torch::Tensor positions,
    torch::Tensor types,
    std::optional<torch::Tensor> cells,
    std::vector<int64_t> indices,
    std::vector<torch::Tensor> fields,
    c10::List<bool> per_atom,
    std::vector<std::string> property_names,
    std::vector<std::optional<torch::Tensor>> forces,
    std::vector<std::optional<torch::Tensor>> stresses
) {
    TORCH_CHECK(na.scalar_type() == torch::kInt64 && na.is_contiguous(), "na must be a contiguous int64 tensor");
    TORCH_CHECK(types.scalar_type() == torch::kInt32 && types.is_contiguous(), "types must be a contiguous int32 tensor");
    auto n_fields = fields.size();
    TORCH_CHECK(per_atom.size() == n_fields && property_names.size() == n_fields &&
        forces.size() == n_fields && stresses.size() == n_fields,
        "every field needs a per_atom flag, a property name, forces and stresses");

    auto batch_size = static_cast<int64_t>(indices.size());
    auto ranges = atom_ranges(na, indices);
    auto structure_count = ones(indices.size());

    // one allocation per array for the whole batch, systems get views into it
    auto all_positions = gather_float(positions, ranges.start, ranges.count, ranges.total);
    auto all_types = torch::empty({ranges.total}, torch::TensorOptions().dtype(torch::kInt32));
    gather_rows<int32_t, int32_t>(types, ranges.start, ranges.count, all_types.data_ptr<int32_t>(), 1);

    torch::Tensor all_cells;
    if (cells.has_value()) {
        all_cells = gather_float(cells.value(), indices, structure_count, batch_size);
    } else {
        all_cells = torch::zeros({batch_size, 3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    }
    // a direction is periodic when its cell vector is non-zero
    auto all_pbcs = all_cells.ne(0.0).any(/*dim=*/2);

    auto systems = std::vector<System>();
    systems.reserve(indices.size());
    int64_t offset = 0;
    for (int64_t k = 0; k < batch_size; k++) {
        auto n = ranges.count[k];
        systems.push_back(torch::make_intrusive<SystemHolder>(
            all_types.narrow(0, offset, n),
            all_positions.narrow(0, offset, n),
            all_cells[k],
            all_pbcs[k]
        ));
        offset += n;
    }

    auto tensors = std::vector<TensorMap>();
    tensors.reserve(n_fields);
    for (size_t f = 0; f < n_fields; f++) {
        const auto& field = fields[f];
        auto properties = LabelsHolder::range(property_names[f], field.size(-1));
        auto components = cartesian_components(field.dim());

        torch::Tensor values;
        Labels samples;
        if (per_atom.get(f)) {
            values = gather_float(field, ranges.start, ranges.count, ranges.total);
            samples = per_atom_samples({"system", "atom"}, indices, ranges);
        } else {
            values = gather_float(field, indices, structure_count, batch_size);
            samples = per_structure_samples("system", indices);
        }
        auto block = torch::make_intrusive<TensorBlockHolder>(values, samples, components, properties);

        auto rows = std::vector<int64_t>(indices.size());
        for (int64_t k = 0; k < batch_size; k++) {
            rows[k] = k;
        }

        if (forces[f].has_value()) {
            // stored as forces, the gradient is their opposite
            auto gradient = gather_float(forces[f].value(), ranges.start, ranges.count, ranges.total, -1.0);
            block->add_gradient("positions", torch::make_intrusive<TensorBlockHolder>(
                gradient,
                per_atom_samples({"sample", "atom"}, rows, ranges),
                cartesian_components(gradient.dim()),
                properties
            ));
        }

        if (stresses[f].has_value()) {
            // stored as stresses, the strain gradient is the virial
            auto stress = gather_float(stresses[f].value(), indices, structure_count, batch_size);
            auto volumes = torch::abs(torch::linalg_det(all_cells));
            block->add_gradient("strain", torch::make_intrusive<TensorBlockHolder>(
                stress * volumes.reshape({batch_size, 1, 1, 1}),
                per_structure_samples("sample", rows),
                cartesian_components(stress.dim()),
                properties
            ));
        }

        tensors.push_back(torch::make_intrusive<TensorMapHolder>(
            LabelsHolder::single(), std::vector<TensorBlock>{block}
        ));
    }

    return {systems, tensors};
}

TORCH_LIBRARY(metatrain_memmap, m) {
    m.def(
        "load_batch("
            "Tensor na, Tensor positions, Tensor types, Tensor? cells, int[] indices, "
            "Tensor[] fields, bool[] per_atom, str[] property_names, "
            "Tensor?[] forces, Tensor?[] stresses"
        ") -> (__torch__.torch.classes.metatomic.System[], "
              "__torch__.torch.classes.metatensor.TensorMap[])",
        load_batch
    );
}

// Loaded as a Python module rather than with torch.ops.load_library, so that
// it stays out of torch.ops.loaded_libraries: metatomic bundles every library
// listed there with exported models, and models never use this op. Loading the
// module runs the TORCH_LIBRARY registration above all the same.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
