#pragma once

#include <cppad/cg/lang/c/lang_c_default_var_name_gen.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Describes a contiguous segment of variables sharing a base name. Lets us
// emit `a[i]`, `b.broadcast(i)`, or `t` (scalar) instead of CppADCG's
// default `x[i]` for every independent or dependent.
struct VarSegment
{
    std::string name;
    std::size_t size;
    bool is_array;
    std::string prefix;
    std::string suffix;

    VarSegment(std::string n, std::size_t s, bool arr)
        : name(std::move(n)), size(s), is_array(arr), prefix("["), suffix("]")
    {
    }

    VarSegment(std::string n, std::size_t s, bool arr, std::string pre, std::string suf)
        : name(std::move(n)), size(s), is_array(arr), prefix(std::move(pre)), suffix(std::move(suf))
    {
    }
};

template <class Base>
class SegmentedVariableNameGenerator : public CppAD::cg::LangCDefaultVariableNameGenerator<Base>
{
public:
    explicit SegmentedVariableNameGenerator(
        std::vector<VarSegment> input_segments,
        const std::string &depName = "y",
        const std::string &tmpName = "v")
        : SegmentedVariableNameGenerator(std::move(input_segments), {}, depName, tmpName)
    {
    }

    SegmentedVariableNameGenerator(
        std::vector<VarSegment> input_segments,
        std::vector<VarSegment> output_segments,
        const std::string &depName = "y",
        const std::string &tmpName = "v")
        : CppAD::cg::LangCDefaultVariableNameGenerator<Base>(depName, "x", tmpName)
        , _input_segments(std::move(input_segments))
        , _output_segments(std::move(output_segments))
    {
        _input_starts.push_back(0);
        for (const auto &seg : _input_segments)
        {
            _input_total += seg.size;
            _input_starts.push_back(_input_total);
        }

        _output_starts.push_back(0);
        for (const auto &seg : _output_segments)
        {
            _output_total += seg.size;
            _output_starts.push_back(_output_total);
        }

        this->_independent.clear();
        for (const auto &seg : _input_segments)
        {
            this->_independent.push_back(CppAD::cg::FuncArgument(seg.name));
        }

        if (not _output_segments.empty())
        {
            this->_dependent.clear();
            for (const auto &seg : _output_segments)
            {
                this->_dependent.push_back(CppAD::cg::FuncArgument(seg.name));
            }
        }
    }

    std::string generateIndependent(const CppAD::cg::OperationNode<Base> &independent, size_t id) override
    {
        _ss.clear();
        _ss.str("");

        std::size_t index = id - 1;
        auto [seg_idx, local_idx] = findInputSegment(index);

        if (seg_idx < _input_segments.size())
        {
            const auto &seg = _input_segments[seg_idx];
            if (seg.is_array)
            {
                _ss << seg.name << seg.prefix << local_idx << seg.suffix;
            }
            else
            {
                _ss << seg.name;
            }
            return _ss.str();
        }
        return CppAD::cg::LangCDefaultVariableNameGenerator<Base>::generateIndependent(independent, id);
    }

    std::string generateDependent(std::size_t index) override
    {
        if (_output_segments.empty())
        {
            return CppAD::cg::LangCDefaultVariableNameGenerator<Base>::generateDependent(index);
        }

        _ss.clear();
        _ss.str("");

        auto [seg_idx, local_idx] = findOutputSegment(index);

        if (seg_idx < _output_segments.size())
        {
            const auto &seg = _output_segments[seg_idx];
            if (seg.is_array)
            {
                _ss << seg.name << seg.prefix << local_idx << seg.suffix;
            }
            else
            {
                _ss << seg.name;
            }
            return _ss.str();
        }
        return CppAD::cg::LangCDefaultVariableNameGenerator<Base>::generateDependent(index);
    }

    const std::string &
    getIndependentArrayName(const CppAD::cg::OperationNode<Base> &indep, size_t id) override
    {
        std::size_t index = id - 1;
        auto [seg_idx, local_idx] = findInputSegment(index);
        if (seg_idx < _input_segments.size())
        {
            return _input_segments[seg_idx].name;
        }
        return CppAD::cg::LangCDefaultVariableNameGenerator<Base>::getIndependentArrayName(indep, id);
    }

    std::size_t
    getIndependentArrayIndex(const CppAD::cg::OperationNode<Base> &indep, std::size_t id) override
    {
        std::size_t index = id - 1;
        auto [seg_idx, local_idx] = findInputSegment(index);
        (void)seg_idx;
        return local_idx;
    }

    bool isConsecutiveInIndepArray(
        const CppAD::cg::OperationNode<Base> &indepFirst,
        std::size_t idFirst,
        const CppAD::cg::OperationNode<Base> &indepSecond,
        std::size_t idSecond) override
    {
        std::size_t i1 = idFirst - 1;
        std::size_t i2 = idSecond - 1;
        auto [seg1, local1] = findInputSegment(i1);
        auto [seg2, local2] = findInputSegment(i2);
        (void)local1;
        (void)local2;
        return seg1 == seg2 and i1 + 1 == i2 and _input_segments[seg1].is_array;
    }

    bool isInSameIndependentArray(
        const CppAD::cg::OperationNode<Base> &indep1,
        std::size_t id1,
        const CppAD::cg::OperationNode<Base> &indep2,
        std::size_t id2) override
    {
        std::size_t i1 = id1 - 1;
        std::size_t i2 = id2 - 1;
        auto [seg1, local1] = findInputSegment(i1);
        auto [seg2, local2] = findInputSegment(i2);
        (void)local1;
        (void)local2;
        return seg1 == seg2;
    }

private:
    std::pair<std::size_t, std::size_t> findInputSegment(std::size_t index) const
    {
        for (std::size_t i = 0; i < _input_segments.size(); ++i)
        {
            if (index < _input_starts[i + 1])
            {
                return {i, index - _input_starts[i]};
            }
        }
        return {_input_segments.size(), 0};
    }

    std::pair<std::size_t, std::size_t> findOutputSegment(std::size_t index) const
    {
        for (std::size_t i = 0; i < _output_segments.size(); ++i)
        {
            if (index < _output_starts[i + 1])
            {
                return {i, index - _output_starts[i]};
            }
        }
        return {_output_segments.size(), 0};
    }

    std::vector<VarSegment> _input_segments;
    std::vector<std::size_t> _input_starts;
    std::size_t _input_total{0};

    std::vector<VarSegment> _output_segments;
    std::vector<std::size_t> _output_starts;
    std::size_t _output_total{0};

    mutable std::stringstream _ss;
};