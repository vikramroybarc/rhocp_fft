#include "DDCPOrientationMap.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
  assert(argc == 2);
  const auto m = DDCPOrientation::Map::read(argv[1]);
  assert((m.grid() == std::array<std::int64_t,3>{{4,4,4}}));
  assert(m.grainCount() == 2);
  assert(m.voxelCount() == 64);
  assert(m.grainId(0, 2, 1) == 1);
  assert(m.grainId(1, 3, 3) == 1);
  assert(m.grainId(2, 0, 0) == 2);
  assert(m.grainId(3, 3, 3) == 2);
  assert(std::abs(m.eulerDegrees(0, 1, 1)[0]) < 1e-12);
  assert(std::abs(m.eulerDegrees(2, 1, 1)[0] - 45.0) < 1e-12);

  // A domain decomposed into two x slabs produces the same global assignment.
  for (int rank = 0; rank < 2; ++rank)
    for (int x = rank*2; x < rank*2+2; ++x)
      for (int y=0; y<4; ++y)
        for (int z=0; z<4; ++z)
          assert(m.grainId(x,y,z) == (rank + 1));

  // Validate that missing files are rejected.
  bool threw = false;
  try { DDCPOrientation::Map::read("/nonexistent/orientation.in"); }
  catch (const std::runtime_error &) { threw = true; }
  assert(threw);
  std::cout << "orientation.in parsing and global voxel mapping: PASS\n";
}
