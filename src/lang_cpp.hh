#pragma once

namespace CppAD
{
    namespace cg
    {

        template <class Base>
        class LanguageCCustom : public LanguageC<Base>
        {
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
        };
    }  // namespace cg
}  // namespace CppAD
