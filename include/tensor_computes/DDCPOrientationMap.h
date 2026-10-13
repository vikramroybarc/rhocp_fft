#pragma once

// Reader for a global grain-orientation and voxel-to-grain assignment file.
// No dependency on MOOSE/Torch, so the format can be unit tested separately.
// Grid order: [x][y][z], with z varying fastest, as in contiguous Torch
// tensors of shape [nx, ny, nz, ...]. All indices are zero-based.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace DDCPOrientation
{
class Map
{
public:
  static Map read(const std::string & path)
  {
    std::ifstream in(path);
    if (!in)
      throw std::runtime_error("Cannot open DDCP orientation file: " + path);

    // Strip comments so whitespace and line breaks in the voxel map are flexible.
    std::vector<std::string> words;
    std::string line;
    while (std::getline(in, line))
    {
      const auto hash = line.find('#');
      const auto slash = line.find("//");
      const auto comment = std::min(hash, slash);
      if (comment != std::string::npos)
        line.erase(comment);
      std::istringstream row(line);
      std::string word;
      while (row >> word)
        words.push_back(word);
    }

    std::size_t pos = 0;
    auto take = [&](const std::string & context) -> std::string {
      if (pos == words.size())
        throw std::runtime_error("Premature end of " + path + " while reading " + context);
      return words[pos++];
    };
    auto expect = [&](const std::string & keyword) {
      const auto got = take(keyword);
      if (got != keyword)
        throw std::runtime_error(path + ": expected '" + keyword + "', found '" + got + "'");
    };
    auto integer = [&](const std::string & context) -> std::int64_t {
      const auto token = take(context);
      std::size_t used = 0;
      try
      {
        const auto result = std::stoll(token, &used);
        if (used != token.size())
          throw std::invalid_argument("extra characters");
        return result;
      }
      catch (const std::exception &)
      {
        throw std::runtime_error(path + ": invalid integer '" + token + "' for " + context);
      }
    };
    auto real = [&](const std::string & context) -> double {
      const auto token = take(context);
      std::size_t used = 0;
      try
      {
        const auto result = std::stod(token, &used);
        if (used != token.size() || !std::isfinite(result))
          throw std::invalid_argument("invalid/NaN/Inf value");
        return result;
      }
      catch (const std::exception &)
      {
        throw std::runtime_error(path + ": invalid real '" + token + "' for " + context);
      }
    };

    Map data;
    expect("GRID");
    std::uint64_t voxels = 1;
    for (unsigned int d = 0; d < 3; ++d)
    {
      const auto n = integer("GRID dimension");
      if (n <= 0)
        throw std::runtime_error(path + ": GRID dimensions must be positive");
      data._dims[d] = n;
      if (static_cast<std::uint64_t>(n) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / voxels)
        throw std::runtime_error(path + ": GRID size overflows size_t");
      voxels *= static_cast<std::uint64_t>(n);
    }

    expect("GRAINS");
    const auto ngrains = integer("GRAINS count");
    if (ngrains <= 0 || static_cast<std::uint64_t>(ngrains) > voxels)
      throw std::runtime_error(path + ": invalid number of grains");
    for (std::int64_t row = 0; row < ngrains; ++row)
    {
      const auto id64 = integer("grain ID");
      if (id64 <= 0 || id64 > std::numeric_limits<int>::max())
        throw std::runtime_error(path + ": grain IDs must be positive 32-bit integers");
      const int id = static_cast<int>(id64);
      const std::array<double, 3> euler{{real("phi1"), real("Phi"), real("phi2")}};
      if (!data._grain_euler.emplace(id, euler).second)
        throw std::runtime_error(path + ": duplicate grain ID " + std::to_string(id));
    }

    expect("VOXELS");
    data._grain_ids.reserve(static_cast<std::size_t>(voxels));
    for (std::uint64_t v = 0; v < voxels; ++v)
    {
      const auto id64 = integer("voxel grain ID at flat index " + std::to_string(v));
      if (id64 <= 0 || id64 > std::numeric_limits<int>::max() ||
          data._grain_euler.count(static_cast<int>(id64)) == 0)
        throw std::runtime_error(path + ": unknown grain ID " + std::to_string(id64) +
                                 " at voxel " + std::to_string(v));
      data._grain_ids.push_back(static_cast<int>(id64));
    }
    if (pos != words.size())
      throw std::runtime_error(path + ": unexpected tokens after voxel map (expected exactly " +
                               std::to_string(voxels) + " voxel IDs)");
    return data;
  }

  const std::array<std::int64_t, 3> & grid() const { return _dims; }
  std::size_t grainCount() const { return _grain_euler.size(); }
  std::size_t voxelCount() const { return _grain_ids.size(); }

  int grainId(std::int64_t i, std::int64_t j, std::int64_t k) const
  {
    return _grain_ids.at(offset(i, j, k));
  }

  std::array<double, 3> eulerDegrees(std::int64_t i,
                                     std::int64_t j,
                                     std::int64_t k) const
  {
    return _grain_euler.at(grainId(i, j, k));
  }

private:
  std::size_t offset(std::int64_t i, std::int64_t j, std::int64_t k) const
  {
    if (i < 0 || j < 0 || k < 0 || i >= _dims[0] || j >= _dims[1] || k >= _dims[2])
      throw std::out_of_range("DDCPOrientation::Map: global voxel index outside GRID");
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(_dims[1]) +
            static_cast<std::size_t>(j)) * static_cast<std::size_t>(_dims[2]) +
           static_cast<std::size_t>(k);
  }

  std::array<std::int64_t, 3> _dims{{0, 0, 0}};
  std::unordered_map<int, std::array<double, 3>> _grain_euler;
  std::vector<int> _grain_ids;
};
} // namespace DDCPOrientation
