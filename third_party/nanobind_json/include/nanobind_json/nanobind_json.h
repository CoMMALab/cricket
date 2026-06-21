/***************************************************************************
* Copyright (c) 2019, Martin Renou, Ian Bell, Gracjan Adamus               *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <string>
#include <set>

#undef snprintf // required to fix an issue with std::snprintf in nlohmann::json
#include <nlohmann/json.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
namespace nl = nlohmann;

namespace nbjson {
    class CircularReferenceError : public std::runtime_error {
    public:
        CircularReferenceError(const char *msg) : std::runtime_error{msg} {}
    };

    template <typename JSONType>
    inline nb::object from_json(const JSONType &j) {
        if (j.is_null()) {
            return nb::none();
        }
        else if (j.is_boolean()) {
            return nb::bool_(j.template get<bool>());
        }
        else if (j.is_number_integer()) {
            return nb::int_(j.template get<long long>());
        }
        else if (j.is_number_float()) {
            return nb::float_(j.template get<double>());
        }
        else if (j.is_string()) {
            return nb::str(j.template get<std::string>().c_str());
        }
        else if (j.is_array()) {
            nb::list obj;
            for (const auto &el : j) {
                obj.append(from_json<JSONType>(el));
            }
            return obj;
        }
        else {
            nb::dict obj;
            for (typename JSONType::const_iterator it = j.cbegin(); it != j.cend(); ++it) {
                obj[nb::str(it.key().c_str())] = from_json<JSONType>(it.value());
            }
            return obj;
        }
    }

    template <typename JSONType>
    inline JSONType to_json(const nb::handle &obj, std::set<const PyObject *> &refs) {
        if (obj.ptr() == nullptr || obj.is_none()) {
            return nullptr;
        }
        else if (nb::isinstance<nb::bool_>(obj)) {
            return nb::cast<bool>(obj);
        }
        else if (nb::isinstance<nb::int_>(obj)) {
            return nb::cast<long long>(obj);
        }
        else if (nb::isinstance<nb::float_>(obj)) {
            return nb::cast<double>(obj);
        }
        else if (nb::isinstance<nb::bytes>(obj)) {
            nb::module_ base64 = nb::module_::import_("base64");
            return nb::cast<std::string>(base64.attr("b64encode")(obj).attr("decode")("utf-8"));
        }
        else if (nb::isinstance<nb::str>(obj)) {
            return nb::cast<std::string>(obj);
        }
        else if (nb::isinstance<nb::tuple>(obj) || nb::isinstance<nb::list>(obj)) {
            auto insert_ret = refs.insert(obj.ptr());
            if (!insert_ret.second) {
                throw CircularReferenceError("Circular reference detected");
            }

            auto out = JSONType::array();
            for (const nb::handle value : obj) {
                out.push_back(to_json<JSONType>(value, refs));
            }

            refs.erase(insert_ret.first);

            return out;
        }
        else if (nb::isinstance<nb::dict>(obj)) {
            auto insert_ret = refs.insert(obj.ptr());
            if (!insert_ret.second) {
                throw CircularReferenceError("Circular reference detected");
            }

            auto out = JSONType::object();
            for (const nb::handle key : obj) {
                out[nb::cast<std::string>(nb::str(key))] = to_json<JSONType>(obj[key], refs);
            }

            refs.erase(insert_ret.first);

            return out;
        }
        else {
            throw std::runtime_error("to_json not implemented for this type of object: " + nb::cast<std::string>(nb::repr(obj)));
        }
    }

    template <typename JSONType>
    inline JSONType to_json(const nb::handle &obj) {
        std::set<const PyObject *> refs;
        return to_json<JSONType>(obj, refs);
    }
} // nbjson

// nlohmann_json serializers
namespace nlohmann {
    #define MAKE_NLJSON_SERIALIZER_DESERIALIZER(T)                  \
    template <>                                                     \
    struct adl_serializer<T> {                                      \
        inline static void to_json(json &j, const T &obj) {         \
            j = nbjson::to_json<json>(obj);                         \
        }                                                           \
                                                                    \
        inline static T from_json(const json &j) {                  \
            return nb::cast<T>(nbjson::from_json<json>(j));         \
        }                                                           \
                                                                    \
        inline static void to_json(ordered_json &j, const T &obj) { \
            j = nbjson::to_json<ordered_json>(obj);                 \
        }                                                           \
                                                                    \
        inline static T from_json(const ordered_json &j) {          \
            return nb::cast<T>(nbjson::from_json<ordered_json>(j)); \
        }                                                           \
    }

    #define MAKE_NLJSON_SERIALIZER_ONLY(T)                          \
    template <>                                                     \
    struct adl_serializer<T> {                                      \
        inline static void to_json(json &j, const T &obj) {         \
            j = nbjson::to_json<json>(obj);                         \
        }                                                           \
                                                                    \
        inline static void to_json(ordered_json &j, const T &obj) { \
            j = nbjson::to_json<ordered_json>(obj);                 \
        }                                                           \
    }

    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::object);

    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::bool_);
    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::int_);
    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::float_);
    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::str);

    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::list);
    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::tuple);
    MAKE_NLJSON_SERIALIZER_DESERIALIZER(nb::dict);

    MAKE_NLJSON_SERIALIZER_ONLY(nb::handle);

    #undef MAKE_NLJSON_SERIALIZER_DESERIALIZER
    #undef MAKE_NLJSON_SERIALIZER_ONLY
} // nlohmann

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// nanobind caster
template <> struct type_caster<nl::json> {
public:
    NB_TYPE_CASTER(nl::json, const_name("JSON"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        try {
            value = nbjson::to_json<nl::json>(src);
            return true;
        }
        catch (nbjson::CircularReferenceError &e) {
            throw e;
        }
        catch (...) {
            return false;
        }
    }

    static handle from_cpp(const nl::json &src, rv_policy policy, cleanup_list *cleanup) noexcept {
        object obj = nbjson::from_json<nl::json>(src);
        return obj.release();
    }
};

template <> struct type_caster<nl::ordered_json> {
public:
    NB_TYPE_CASTER(nl::ordered_json, const_name("OrderedJSON"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        try {
            value = nbjson::to_json<nl::ordered_json>(src);
            return true;
        }
        catch (nbjson::CircularReferenceError &e) {
            throw e;
        }
        catch (...) {
            return false;
        }
    }

    static handle from_cpp(const nl::ordered_json &src, rv_policy policy, cleanup_list *cleanup) noexcept {
        object obj = nbjson::from_json<nl::ordered_json>(src);
        return obj.release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)