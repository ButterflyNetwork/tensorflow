//
//  IOHelper.hpp
//  mps-test
//
//  Created by sschaetz on 3/11/17.
//  Copyright © 2017 BNI. All rights reserved.
//

#ifndef IOHelper_h
#define IOHelper_h

#include <fstream>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/tensor.h"

namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        inline explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }

        inline int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return ifs_.gcount();
        }

    private:
        std::ifstream ifs_;
    };
}  // namespace


inline NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path =
    [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

inline bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                             new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

std::string getTestdataPath()
{
    NSString *resourcePath = [[NSBundle mainBundle] resourcePath];
    NSString *relativePath = [@[@"/Library/Application Support",
                                [[NSBundle mainBundle] objectForInfoDictionaryKey:@"CFBundleName"],
                                @"testdata"]
                              componentsJoinedByString:@"/"];

    return [[resourcePath stringByAppendingString: relativePath] cStringUsingEncoding:NSASCIIStringEncoding];
}

inline bool checkIfFileExists(const std::string& f)
{
    NSFileManager *fileManager = [NSFileManager defaultManager];

    if ([fileManager fileExistsAtPath:[NSString stringWithUTF8String:f.c_str()]])
    {
        return true;
    }
    else
    {
        return false;
    }
}

#endif /* IOHelper_h */
