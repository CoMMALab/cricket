#pragma once

#include "lang_cpp.hh"

#include <cppad/cg/lang/c/language_c.hpp>

namespace CppAD
{
    namespace cg
    {
        template <class Base>
        class LanguageCVampBlock : public LanguageCCustom<Base>
        {
        public:
            explicit LanguageCVampBlock(std::string varTypeName, size_t spaces = 3)
              : LanguageCCustom<Base>(std::move(varTypeName), spaces)
            {
            }

        protected:
            using Node = typename LanguageCCustom<Base>::Node;
            using Arg = typename LanguageCCustom<Base>::Arg;

            void pushConditionalAssignment(Node &node) override
            {
                const auto &args = node.getArguments();
                const Arg &left = args[0];
                const Arg &right = args[1];
                const Arg &trueCase = args[2];
                const Arg &falseCase = args[3];

                const bool isDep = this->isDependent(node);
                const std::string &varName = this->createVariableName(node);

                if ((trueCase.getParameter() != nullptr and falseCase.getParameter() != nullptr and
                     *trueCase.getParameter() == *falseCase.getParameter()) or
                    (trueCase.getOperation() != nullptr and falseCase.getOperation() != nullptr and
                     trueCase.getOperation() == falseCase.getOperation()))
                {
                    this->pushAssignmentStart(node, varName, isDep);
                    this->push(trueCase);
                    this->pushAssignmentEnd(node);
                    return;
                }

                auto push_wrapped = [&](const Arg &arg)
                {
                    this->_streamStack << "V(";
                    this->push(arg);
                    this->_streamStack << ")";
                };

                this->pushAssignmentStart(node, varName, isDep);
                push_wrapped(falseCase);
                this->_streamStack << ".blend(";
                push_wrapped(trueCase);
                this->_streamStack << ", (";
                push_wrapped(left);
                this->_streamStack << " " << this->getComparison(node.getOperationType()) << " ";
                push_wrapped(right);
                this->_streamStack << "))";
                this->pushAssignmentEnd(node);
            }
        };
    }  // namespace cg
}  // namespace CppAD

