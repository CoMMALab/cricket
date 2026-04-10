#pragma once

namespace CppAD
{
    namespace cg
    {

        template <class Base>
        class LanguageCCustom : public LanguageC<Base>
        {
        private:
            size_t _maskNumber = 0;

        public:
            explicit LanguageCCustom(std::string varTypeName, size_t spaces = 3)
              : LanguageC<Base>(varTypeName, spaces)
            {
            }

            virtual void printParameter(const Base &value)
            {
                writeParameter(value, LanguageC<Base>::_code);
            }

            virtual void pushParameter(const Base &value)
            {
                writeParameter(value, LanguageC<Base>::_streamStack);
            }

            template <class Output>
            void writeParameter(const Base &value, Output &output)
            {
                // make sure all digits of floating point values are printed
                std::ostringstream os;
                os << std::setprecision(LanguageC<Base>::_parameterPrecision) << value;

                std::string number = os.str();
                output << number;

                if (number.find('.') == std::string::npos && number.find('e') == std::string::npos)
                {
                    // also make sure there is always a '.' after the number in
                    // order to avoid integer overflows
                    output << '.';
                }
            }

            // Override to generate SIMD-friendly blend operations instead of if/else
            void pushConditionalAssignment(typename LanguageC<Base>::Node &node) override
            {
                using Arg = typename LanguageC<Base>::Arg;

                CPPADCG_ASSERT_UNKNOWN(this->getVariableID(node) > 0)

                const std::vector<Arg> &args = node.getArguments();
                const Arg &left = args[0];
                const Arg &right = args[1];
                const Arg &trueCase = args[2];
                const Arg &falseCase = args[3];

                const std::string &varName = this->createVariableName(node);

                // Check if true and false cases are the same (optimization)
                if ((trueCase.getParameter() != nullptr && falseCase.getParameter() != nullptr &&
                     *trueCase.getParameter() == *falseCase.getParameter()) ||
                    (trueCase.getOperation() != nullptr && falseCase.getOperation() != nullptr &&
                     trueCase.getOperation() == falseCase.getOperation()))
                {
                    // true and false cases are the same - no conditional needed
                    this->pushAssignmentStart(node, varName, this->isDependent(node));
                    this->push(trueCase);
                    this->pushAssignmentEnd(node);
                    return;
                }

                std::string maskName = "mask_" + std::to_string(_maskNumber++);

                // auto mask_N = left comparison right;
                this->_streamStack << this->_indentation << "auto " << maskName << " = ";
                this->push(left);
                this->_streamStack << " " << this->getComparison(node.getOperationType()) << " ";
                this->push(right);
                this->_streamStack << ";\n";

                // varName = varName.blend(trueCase, mask_N).blend(falseCase, ~mask_N);
                this->_streamStack << this->_indentation << varName << " = " << varName << ".blend(";
                this->push(trueCase);
                this->_streamStack << ", " << maskName << ").blend(";
                this->push(falseCase);
                this->_streamStack << ", ~" << maskName << ");\n";
            }
        };
    }  // namespace cg
}  // namespace CppAD
