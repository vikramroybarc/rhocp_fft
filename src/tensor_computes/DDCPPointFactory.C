#include "DDCPPointFactory.h"
#include "DDCP4P2PointModified.h"

#include <map>
#include <stdexcept>
#include <utility>

namespace
{
using DDCPPointRegistry = std::map<std::string, DDCPPointFactory::Creator>;
DDCPPointRegistry &
registry()
{
  // Explicit built-ins avoid static-initialization-order and linker
  // dead-stripping issues for these two material point implementations.
  static DDCPPointRegistry models = {
    {"DDCP4P2Point",
      [](const DDCP4P2Data & data, const DDCPPointOptions & options)
      {
        if (options.stress_scale != 1.0)
          throw std::invalid_argument(
              "model_stress_scale is only applicable to DDCP4P2PointModified");
        return std::make_unique<DDCP4P2Point>(data);
      }},
    {"DDCP4P2PointModified",
      [](const DDCP4P2Data & data, const DDCPPointOptions & options)
      {
        return std::make_unique<DDCP4P2PointModified>(data, options.stress_scale);
      }}
  };
  return models;
}
} // namespace

void
DDCPPointFactory::registerModel(const std::string & name, Creator creator)
{
  if (name.empty() || !creator)
    throw std::invalid_argument("DDCP point registration requires name and creator");
  if (!registry().emplace(name, std::move(creator)).second)
    throw std::invalid_argument("Duplicate DDCP point model registration: " + name);
}

std::vector<std::string>
DDCPPointFactory::availableModels()
{
  std::vector<std::string> names;
  for (const auto & entry : registry())
    names.push_back(entry.first);
  return names;
}

std::unique_ptr<DDCP4P2Point>
DDCPPointFactory::create(const std::string & name,
                         const DDCP4P2Data & data,
                         const DDCPPointOptions & options)
{
  const auto it = registry().find(name);
  if (it == registry().end())
  {
    std::string message = "Unknown material_model '" + name + "'. Available:";
    for (const auto & candidate : availableModels())
      message += " " + candidate;
    throw std::invalid_argument(message);
  }
  auto point = it->second(data, options);
  if (!point || point->stateSize() == 0)
    throw std::runtime_error("Model '" + name + "' returned a null point or zero stateSize()");
  return point;
}
