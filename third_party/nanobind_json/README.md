# nanobind_json

`nanobind_json` is an `nlohmann::json` to `nanobind` bridge, it allows you to automatically convert `nlohmann::json` to `nb::object` and the other way around. Simply include the header, and the automatic conversion will be enabled.

The library has been tested on Linux, macOS, Windows and Linux-ARM on Python versions >3.8 with the help of Github Actions.

![](https://img.shields.io/github/actions/workflow/status/Griger5/nanobind_json/testing.yml?logo=Github%20Actions&label=Tests) [![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux) [![Linux-ARM OK](https://img.shields.io/static/v1?label=Linux-ARM&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux) [![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS) [![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=blue&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)

> [!TIP]
> Looking for an equivalent with pybind11? Check this out https://github.com/pybind/pybind11_json !

## Dependencies

``nanobind_json`` depends on

- [nanobind](https://github.com/wjakob/nanobind)
- [nlohmann_json](https://github.com/nlohmann/json)

| `nanobind_json` | `nlohmann_json` | `nanobind` |
| ----------------- | ----------------- | ------------ |
| master            | >=3.2.0           | >=2.2.0      |

## C++ API: Automatic conversion between `nlohmann::json` and `nanobind` Python objects

```cpp
#include "nanobind_json/nanobind_json.h"

namespace nb = nanobind;
namespace nl = nlohmann;

nb::dict obj;
obj["number"] = 1234;
obj["hello"] = "world";

// Automatic nb::dict->nl::json conversion
nl::json j = obj;

// Automatic nl::json->nb::object conversion
nb::object result1 = j;
// Automatic nl::json->nb::dict conversion
nb::dict result2 = j;
```

## Making bindings

You can easily make bindings for C++ classes/functions that make use of `nlohmann::json`.

For example, making a binding for the following two functions is automatic, thanks to `nanobind_json`:

```cpp
void take_json(const nlohmann::json &json) {
    std::cout << "This function took an nlohmann::json instance as argument: " << json << std::endl;
}

nlohmann::json return_json() {
    nlohmann::json j = {{"value", 1}};

    std::cout << "This function returns an nlohmann::json instance: "  << j << std::endl;

    return j;
}
```

Bindings:

```cpp
NB_MODULE(my_module, m) {
    m.doc() = "My awesome module";

    m.def("take_json", &take_json, "pass nb::object to a C++ function that takes an nlohmann::json");
    m.def("return_json", &return_json, "return nb::object from a C++ function that returns an nlohmann::json");
}
```

You can then use your functions from Python:

```python
import my_module

my_module.take_json({"value": 2})
j = my_module.return_json()

print(j)
```

## Using the library in your project

Since the library is header-only, the easiest way to use it in your project is to simply download the "nanobind_json.h" file into your project directory. Another way is adding `nanobind_json` as a **git submodule**.

*NOTE: You need both `nlohmann_json` and `nanobind` also available in your project. It is recommended to add all three libraries as git submodules*.

## Tests

To build and run the tests, simply do the following:

```bash
cmake -S . -B build/ -DNBJSON_TEST=1
cmake --build build/
cd build/tests/
python -m pytest -s -vv ../../tests/
```

If `nanobind` and `nlohmann_json` CMake packages are not found they will be downloaded through CMake's `FetchContent`.

## License

We use a shared copyright model that enables all contributors to maintain the copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
