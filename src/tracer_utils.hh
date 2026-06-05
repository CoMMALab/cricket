#pragma once 

#include <fmt/core.h>
#include <nlohmann/json.hpp>

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

void add_to_trace(Traced tcode, std::string name, nlohmann::json &data)
{
    data[name] = tcode.code;
    data[fmt::format("{}_vars", name)] = tcode.temp_variables;
    data[fmt::format("{}_output", name)] = tcode.outputs;
}

