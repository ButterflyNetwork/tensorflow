# pragma once

#import <Foundation/Foundation.h>

#include <numeric>
#include <functional>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

namespace numpyhelper
{

/// @brief Numpy specific type traits for floating point types (map C++ types to numpy type).
/// @return 'f' If T is floating point number.
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, char>::type
MapTypeTrait()
{
    return 'f';
}

/// @brief Numpy specific type traits for signed integer types (map C++ types to numpy type).
/// @return 'i' If T is signed integer.
template <typename T>
inline typename std::enable_if<std::is_signed<T>::value && std::is_integral<T>::value, char>::type
MapTypeTrait()
{
    return 'i';
}

/// @brief Numpy specific type traits for unsigned integer types (map C++ types to numpy type).
/// @return 'u' If T is unsigned integer.
template <typename T>
inline typename std::enable_if<std::is_unsigned<T>::value && std::is_integral<T>::value, char>::type
MapTypeTrait()
{
    return 'u';
}

/// @brief Numpy specific endian trait.
/// @return '>' for big endian, '<' for little endian.
inline char EndiannessTrait()
{
    /// @todo: this does not really exist.
#ifdef TENSORFLOW_IS_BIG_ENDIAN
        return '>';
#else
        return '<';
#endif
}

/// @brief Load an array from a stream.
/// @tparam T Type of elements that are read from stream.
/// @tparam Stream Type of stream that numpy file is read from.
/// @param stream Stream the numpy file is read from.
/// @return Tuple containing 2 flat vectors: the first vector contains bounds of data, the second contains data.
template <typename T, typename Stream>
std::tuple <std::vector<std::size_t>, std::vector<T>> ReadNumpy(Stream& stream)
{
    std::tuple <std::vector<std::size_t>, std::vector<T>> empty;
    std::vector<std::size_t> bounds;
    unsigned int wordsize;

    // Parse the header.
    {
        char buffer[8];
        stream.read(buffer, 8);

        if (!stream)
        {
            NSLog(@"Failed reading first 8 bytes of npy header.");
            return empty;
        }

        uint16_t headersize = 0;
        stream.read(reinterpret_cast<char*>(&headersize), 2);

        if (!stream)
        {
            NSLog(@"Failed Reading the header bytes..");
            return empty;
        }

        std::vector<char> headerbuffer(headersize);
        stream.read(&headerbuffer[0], headersize);

        if (!stream)
        {
            NSLog(@"Failed reading npy header.");
            return empty;
        }

        std::string header(&headerbuffer[0], headersize);
        int loc1, loc2;

        // Fortran order.
        loc1 = header.find("fortran_order") + 16;

#if 0
        // TODO: This variable was here but unused.
        bool fortranOrder = (header.substr(loc1, 5) == "True" ? true : false);
#endif

        // Bounds.
        loc1 = header.find("(");
        loc2 = header.find(")");
        std::string str_shape = header.substr(loc1+1, loc2-loc1-1);

        if (str_shape[str_shape.size()-1] == ',')
        {
            bounds.resize(1);
        }
        else
        {
            bounds.resize(std::count(str_shape.begin(), str_shape.end(), ',') + 1);
        }

        for (unsigned int i = 0;i < bounds.size(); i++)
        {
            loc1 = str_shape.find(",");
            bounds[i] = atoi(str_shape.substr(0, loc1).c_str());
            str_shape = str_shape.substr(loc1+1);
        }

        loc1 = header.find("descr") + 9;
        bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);

        if (!littleEndian)
        {
            NSLog(@"Little endian expected, but is not little endian.");
            return empty;
        }

        std::string str_ws = header.substr(loc1+2);
        loc2 = str_ws.find("'");
        wordsize = atoi(str_ws.substr(0, loc2).c_str());
        if (wordsize != sizeof(T))
        {
            NSLog(@"Word size does not match expected type.");
        }
    }
    
    std::size_t payloadSize = std::accumulate(bounds.begin(), bounds.end(), 1, std::multiplies<std::size_t>());
    if (payloadSize > 0)
    {
        std::vector<T> data(payloadSize);
        stream.read(reinterpret_cast<char*>(data.data()), sizeof(T)*payloadSize);
        return std::make_tuple(bounds, data);
    }
    else
    {
        // Return empty array if the bounds are zero.
        return empty;
    }
}

/// @brief Load an array from a file.
/// @tparam T Type of elements that are read from stream.
/// @return Tuple containing 2 flat vectors: the first vector contains bounds of data, the second contains data.
template <typename T>
std::tuple <std::vector<std::size_t>, std::vector<T>> ReadNumpyFile(const std::string& filename)
{
    std::ifstream s(filename, std::ios::binary);
    return ReadNumpy<T>(s);
}

/// @brief Load an array from a file.
/// @tparam T Type of elements that are read from stream.
/// @return Tuple containing 2 flat vectors: the first vector contains bounds of data, the second contains data.
template <typename T>
std::tuple <std::vector<std::size_t>, std::vector<T>> ReadNumpyFile(NSString* filename)
{
    std::ifstream s(std::string([filename UTF8String]), std::ios::binary);
    return ReadNumpy<T>(s);
}

// ---------------------------------------------------------------------------------------------------------------------
} // numpyhelper
// ---------------------------------------------------------------------------------------------------------------------
