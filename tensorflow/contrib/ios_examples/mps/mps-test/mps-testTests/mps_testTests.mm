//
//  mps_testTests.m
//  mps-testTests
//
//  Created by sschaetz on 3/20/17.
//  Copyright © 2017 BNI. All rights reserved.
//

#import <XCTest/XCTest.h>

#import <Foundation/Foundation.h>

#include <tuple>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"


#include "IOHelper.h"
#import "NumpyHelper.hpp"

std::tuple<std::string, std::string, std::string> getTestFilePaths(int test_file_id)
{
    std::string image_input_file, filter_input_file, expected_result_file;
    image_input_file =
        getTestdataPath() +
        std::string("/conv2d_image_input") +
        std::to_string(test_file_id) +
        std::string(".npy");
    filter_input_file =
        getTestdataPath() +
        std::string("/conv2d_filter_input") +
        std::to_string(test_file_id) +
        std::string(".npy");
    expected_result_file =
        getTestdataPath() +
        std::string("/conv2d_expected_result") +
        std::to_string(test_file_id) +
        std::string(".npy");
    return std::make_tuple(image_input_file, filter_input_file, expected_result_file);
}

bool checkTestFilePaths(const std::tuple<std::string, std::string, std::string>& paths)
{
    return
        checkIfFileExists(std::get<0>(paths)) &&
        checkIfFileExists(std::get<1>(paths)) &&
        checkIfFileExists(std::get<2>(paths));
}

@interface mps_testTests : XCTestCase

@end

@implementation mps_testTests

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

// This test makes sure that we can read numpy files correctly.
- (void)testNumpyIO {
    auto file = getTestdataPath() + std::string("/test_numpy_input.npy");
    auto npyArray = numpyhelper::ReadNumpyFile<float>(file);

    std::vector<std::size_t> expectedBounds = { 2, 4, 2 };
    std::vector<float> expectedVector = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f
    };

    XCTAssert(std::equal(expectedBounds.begin(), expectedBounds.end(), std::get<0>(npyArray).begin()));
    XCTAssert(std::equal(expectedVector.begin(), expectedVector.end(), std::get<1>(npyArray).begin()));
}

// This tests runs the conv2d operation on all inputs that are found.
- (void)testConv2D {
    // Load graph.
    tensorflow::SessionOptions options;

    tensorflow::Session* session_pointer = nullptr;
    auto session_status = tensorflow::NewSession(options, &session_pointer);
    XCTAssert(session_status.ok());

    std::unique_ptr<tensorflow::Session> session(session_pointer);
    tensorflow::GraphDef tensorflow_graph;

    const bool read_proto_succeeded = PortableReadFileToProto(
        getTestdataPath() + std::string("/conv2d_graph.pb"),
        &tensorflow_graph
    );
    XCTAssert(read_proto_succeeded);
    tensorflow::Status s = session->Create(tensorflow_graph);
    XCTAssert(s.ok());

    int test_file_id = 0;
    auto test_files = getTestFilePaths(test_file_id);
    while (checkTestFilePaths(test_files))
    {
        auto imageInput = numpyhelper::ReadNumpyFile<float>(std::get<0>(test_files));
        auto imageInputBounds = std::get<0>(imageInput);
        auto imageInputVector = std::get<1>(imageInput);

        auto filterInput = numpyhelper::ReadNumpyFile<float>(std::get<1>(test_files));
        auto filterInputBounds = std::get<0>(filterInput);
        auto filterInputVector = std::get<1>(filterInput);

        auto goldOutput = numpyhelper::ReadNumpyFile<float>(std::get<2>(test_files));

        tensorflow::Tensor imageTensor(
           tensorflow::DT_FLOAT,
           tensorflow::TensorShape(
               {
                   static_cast<long long>(imageInputBounds[0]),
                   static_cast<long long>(imageInputBounds[1]),
                   static_cast<long long>(imageInputBounds[2]),
                   static_cast<long long>(imageInputBounds[3])
               })
           );
        auto imageTensorMapped = imageTensor.tensor<float, 4>();
        std::copy(imageInputVector.begin(), imageInputVector.end(), imageTensorMapped.data());

        tensorflow::Tensor filterTensor(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape(
                {
                    static_cast<long long>(filterInputBounds[0]),
                    static_cast<long long>(filterInputBounds[1]),
                    static_cast<long long>(filterInputBounds[2]),
                    static_cast<long long>(filterInputBounds[3])
                })
            );
        auto filterTensorMapped = filterTensor.tensor<float, 4>();
        std::copy(filterInputVector.begin(), filterInputVector.end(), filterTensorMapped.data());
        
        
        tensorflow::string imageInputStr = "image_in";
        tensorflow::string filterInputStr = "filter_in";
        tensorflow::string filteredOutputStr = "filtered_output";
        std::vector<tensorflow::Tensor> outputs;
        
        // Run graph.
        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
        tensorflow::RunMetadata run_metadata;
        auto run_status = session->Run(
             run_options,
             {
                 {imageInputStr, imageTensor},
                 {filterInputStr, filterTensor}
             },
             {filteredOutputStr},
             {},
             &outputs,
             &run_metadata
        );
        XCTAssert(run_status.ok());

#ifdef TENSORFLOW_MPS_PRINT_PERF
        const tensorflow::StepStats& step_stats = run_metadata.step_stats();
        auto statSummarizer = std::make_shared<tensorflow::StatSummarizer>(tensorflow_graph);
        statSummarizer->ProcessStepStats(step_stats);
        std::cout << statSummarizer->GetOutputString() << std::endl;
#endif

        // Compare output dimensions.
        XCTAssert(
            std::equal(
               std::get<0>(goldOutput).begin(),
               std::get<0>(goldOutput).end(),
               outputs[0].shape().begin(),
               [](const std::size_t &a, const tensorflow::TensorShapeDim &b)
               {
                   return static_cast<long long>(a) == b.size;
               }
            )
        );

        // Compare contents of output.
        XCTAssert(
            std::equal(
                std::get<1>(goldOutput).begin(),
                std::get<1>(goldOutput).end(),
                outputs[0].tensor<float, 4>().data(),
                [](float a, float b)
                {
                    return fabs(a - b) < std::numeric_limits<float>::epsilon();
                }
            )
        );

        // Load data for next test.
        test_file_id += 1;
        test_files = getTestFilePaths(test_file_id);
    }
    // At least one test must run.
    XCTAssert(test_file_id > 0);
}



@end
