#pragma once

#include <iomanip>
#include "cppad/cg/lang/c/language_c.hpp"

namespace CppAD
{
    namespace cg
    {

        template <class Base>
        class LanguageRust : public LanguageC<Base>
        {
        public:
            explicit LanguageRust(std::string varTypeName, size_t spaces = 3)
              : LanguageC<Base>(varTypeName, spaces)
            {
            }

            virtual void printParameter(const Base &value)
            {
                writeParameter(value, LanguageRust<Base>::_code);
            }

            virtual void pushParameter(const Base &value)
            {
                writeParameter(value, LanguageRust<Base>::_streamStack);
            }

            template <class Output>
            void writeParameter(const Base &value, Output &output)
            {
                // make sure all digits of floating point values are printed
                std::ostringstream os;
                os << std::setprecision(LanguageRust<Base>::_parameterPrecision) << value;

                std::string number = os.str();
                output << "Simd::<f32, L>::splat(";
                output << number;

                if (number.find('.') == std::string::npos && number.find('e') == std::string::npos)
                {
                    // also make sure there is always a '.' after the number in
                    // order to avoid integer overflows
                    output << '.';
                }
                output << ")";
            }
        };
    }  // namespace cg
}  // namespace CppAD
