#pragma once

// Lightweight C++ factory for DDCP4P2Point and its subclasses.
// These material points are not MOOSE objects and do not use registerMooseObject.
#include "DDCP4P2Point.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct DDCPPointOptions
{
  // Illustrative only; used by DDCP4P2PointModified, not the baseline model.
  Real stress_scale = 1.0;
};

class DDCPPointFactory
{
public:
  using Creator = std::function<std::unique_ptr<DDCP4P2Point>(
      const DDCP4P2Data &, const DDCPPointOptions &)>;

  // Extensions register ordinary DDCP4P2Point subclasses by name.
  static void registerModel(const std::string & name, Creator creator);

  static std::unique_ptr<DDCP4P2Point>
  create(const std::string & name,
         const DDCP4P2Data & data,
         const DDCPPointOptions & options);

  static std::vector<std::string> availableModels();
};
