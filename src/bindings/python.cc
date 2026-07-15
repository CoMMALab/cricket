#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nanobind_json/nanobind_json.h>

#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core_ext, m)
{
    m.doc() = "Cricket: tracing compilation for spherized robot kinematics.";

    nb::class_<cricket::RobotInfo>(m, "RobotInfo")
        .def(
            "__init__",
            [](cricket::RobotInfo *self,
               const std::filesystem::path &urdf,
               const std::optional<std::filesystem::path> &srdf,
               const std::optional<std::string> &end_effector)
            { new (self) cricket::RobotInfo(urdf, srdf, end_effector); },
            "urdf"_a,
            "srdf"_a = nb::none(),
            "end_effector"_a = nb::none(),
            "Parse a robot URDF.")
        .def_prop_ro("dimension", [](cricket::RobotInfo &r) { return r.model.nq; })
        .def_prop_ro("n_spheres", [](cricket::RobotInfo &r) { return r.spheres.size(); })
        .def_prop_ro("min_radius", [](cricket::RobotInfo &r) { return r.min_radius; })
        .def_prop_ro("max_radius", [](cricket::RobotInfo &r) { return r.max_radius; })
        .def_prop_ro("end_effector_name", [](cricket::RobotInfo &r) { return r.end_effector_name; })
        .def(
            "json",
            [](cricket::RobotInfo &r) { return nbjson::from_json<nlohmann::json>(r.json()); },
            "Return the parsed robot's metadata.");

    nb::class_<cricket::GenOptions>(m, "GenOptions")
        .def(
            "__init__",
            [](cricket::GenOptions *self,
               const std::filesystem::path &urdf,
               const std::optional<std::filesystem::path> &srdf,
               nb::object end_effector,
               const std::optional<std::filesystem::path> &template_path,
               const std::map<std::string, std::filesystem::path> &subtemplates,
               const std::string &language,
               nb::object data)
            {
                new (self) cricket::GenOptions{};
                self->urdf = urdf;
                self->srdf = srdf;
                // GenOptions carries multiple end effectors; accept a single str or a list.
                if (not end_effector.is_none())
                {
                    if (nb::isinstance<nb::str>(end_effector))
                    {
                        auto s = nb::cast<std::string>(end_effector);
                        if (not s.empty())
                        {
                            self->end_effectors.push_back(std::move(s));
                        }
                    }
                    else
                    {
                        self->end_effectors = nb::cast<std::vector<std::string>>(end_effector);
                    }
                }
                if (template_path)
                {
                    self->template_path = *template_path;
                }
                self->subtemplates = subtemplates;
                self->language = language;
                if (not data.is_none())
                {
                    self->data = nb::cast<nlohmann::json>(data);
                }
            },
            "urdf"_a,
            "srdf"_a = nb::none(),
            "end_effector"_a = nb::none(),
            "template_path"_a = nb::none(),
            "subtemplates"_a = std::map<std::string, std::filesystem::path>{},
            "language"_a = std::string("c++"),
            "data"_a = nb::dict())
        .def_rw("urdf", &cricket::GenOptions::urdf)
        .def_rw("srdf", &cricket::GenOptions::srdf)
        .def_rw("end_effectors", &cricket::GenOptions::end_effectors)
        .def_rw("template_path", &cricket::GenOptions::template_path)
        .def_rw("subtemplates", &cricket::GenOptions::subtemplates)
        .def_rw("language", &cricket::GenOptions::language);

    nb::class_<cricket::GenResult>(m, "GenResult")
        .def_ro("source", &cricket::GenResult::source)
        .def_ro("robot_name", &cricket::GenResult::robot_name)
        .def_ro("dimension", &cricket::GenResult::dimension)
        .def_ro("n_spheres", &cricket::GenResult::n_spheres)
        .def_prop_ro(
            "data", [](cricket::GenResult &r) { return nbjson::from_json<nlohmann::json>(r.data); });

    m.def(
        "generate_robot_source",
        [](const cricket::GenOptions &opts) { return cricket::generate_robot_source(opts); },
        "opts"_a,
        "Run the URDF → traced FK/CC code → inja template render pipeline; return a "
        "GenResult with the rendered C++ source + metadata.");
}
