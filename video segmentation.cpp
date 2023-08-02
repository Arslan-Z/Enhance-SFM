// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
/*
��������ͼ�񼯣���ȡ������ƥ�������㣬�õ�ÿ��ͼ���ĳ�ʼƥ���ԣ�
ʹ��RANSAC�㷨�ӳ�ʼƥ�����й��ƻ�������
���ݻ����������ÿ��ͼ�������λ�ˣ�
��������ָ��Ŀ¼��
���ply
*/
#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/svg_features.hpp"
#include "openMVG/geometry/pose3.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/image_concat.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching/svg_matches.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/pipelines/sfm_robust_model_estimation.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"


#include <iostream>
#include <string>
#include <utility>
#include <filesystem>

#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <chrono>


namespace fs = std::filesystem;

using namespace openMVG;
using namespace openMVG::matching;
using namespace openMVG::image;
using namespace openMVG::cameras;
using namespace openMVG::geometry;

/// Read intrinsic K matrix from a file (ASCII)
/// F 0 ppx
/// 0 F ppy
/// 0 0 1
bool readIntrinsic(const std::string& fileName, Mat3& K);//�����ں���

/// Export 3D point vector and camera position to PLY format�� 3D ��ʸ�������λ�õ���Ϊ PLY ��ʽ�������ں���
bool exportToPly(const std::vector<Vec3>& vec_points,
    const std::vector<Vec3>& vec_camPos,
    const std::string& sFileName);


void processFrame(cv::Mat frame, int frameNumber, const std::string& outputPath) {
    // Process the frame (e.g., extract EXIF data, save the frame to an image file, etc.)
    // ...

    // Save the frame to an image f ile
    std::stringstream ss;
    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber << ".jpg";
    cv::imwrite(ss.str(), frame);

    // Display the frame number
    std::cout << "Processing frame " << frameNumber << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <video_path> <output_path> <interval>" << std::endl;  //�������
        return -1;
    }

    std::string videoPath = argv[1];
    std::string outputPath = argv[2];
    long int interval = std::stoi(argv[3]);


    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }


    // Get the video frame rate
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Frame rate: " << fps << " FPS" << std::endl;

    // Create a vector of threads to process the frames
   // std::vector<std::thread> threads;




    // Process the video frames
    cv::Mat frame;
    long int frameNumber = 0;  //��������ڽ�ȡ��֡����
    long int nummm = 1 * interval;//�Ȼ���2��
    long int frameNumber1 = 0; //���ڽ��д����֡��
    long int frameNumber2 = interval;
    long int numm11 = interval;
    long int modo1 = 1; //ģʽѡ�� //1���Ǵ�ǰ����0���ǴӺ�
    long int maxx = 0; //���ֵȷ��


    while (frameNumber <= nummm)
    {
        // Read the next frame

        while (numm11--)
        {
            if (numm11 == (interval - 1))
            {
                cap >> frame; // ��ȡ֡
            }
            else {
                cap.grab(); // ����֡
            }

            if (frame.empty()) break; // ���û�и���֡�����˳�ѭ��
        }
        numm11 = interval;
        if (frame.empty()) break; // ���û�и���֡�����˳�ѭ��
        // Only process frames with interval
        if (frameNumber % interval == 0) {
            // Start a new thread to process the current frame
            //threads.emplace_back(processFrame, frame.clone(), frameNumber, outputPath);
            processFrame(frame.clone(), frameNumber, outputPath);

        }

        frameNumber = frameNumber + interval;
    }
    ////

    {
        std::stringstream ss;
        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber1 << ".jpg";
        const std::string jpg_filenameL = ss.str();
        ss.str(""); ss.clear();
        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber2 << ".jpg";
        const std::string jpg_filenameR = ss.str();
        // ������ʹ�� jpg_filenameL ����
        ss.str(""); ss.clear();


        Image<unsigned char> imageL, imageR;   //��ͼƬ
        ReadImage(jpg_filenameL.c_str(), &imageL);
        ReadImage(jpg_filenameR.c_str(), &imageR);
        /* int maxAttempts = 20;
         int attempts = 0;
         bool successL = false, successR = false;
         std::chrono::milliseconds waitTime(10); // �ȴ� 5 ����

         while (attempts < maxAttempts && (!successL || !successR)) {
             if (!successL) {
                 successL = ReadImage(jpg_filenameL.c_str(), &imageL);
             }
             if (!successR) {
                 successR = ReadImage(jpg_filenameR.c_str(), &imageR);
             }

             if (!successL || !successR) {
                 attempts++;
                 std::cout << "Attempt " << attempts << " failed. Waiting for 10m second before retrying..." << std::endl;
                 std::this_thread::sleep_for(waitTime);
             }
         }
         */
         //--
         // Detect regions thanks to an image_describer��������򣬸���ͼ��������
         //--
        using namespace openMVG::features;
        std::unique_ptr<Image_describer> image_describer(new SIFT_Anatomy_Image_describer); //��������������
        std::map<IndexT, std::unique_ptr<features::Regions>> regions_perImage;     //������һ��std::map�������ڴ洢ÿ��ͼ�����������������
                                            //����������������һ�����ڱ�ʾͼ��ֲ������ķ�������ͨ������������ɣ������������������������
        image_describer->Describe(imageL, regions_perImage[0]);   //��imageLͼ�����������ȡ���������������㣬��������洢��regions_perImage[0]��
        image_describer->Describe(imageR, regions_perImage[1]);    //�������������ȡ��

        const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
        const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());//��ͼ��0��ͼ��1����������������ת��ΪSIFT_Regions���ͣ����ֱ�洢��regionsL��regionsR��

        const PointFeatures
            featsL = regions_perImage.at(0)->GetRegionsPositions(),  //����λ����ȡ��������Ϊ���������������
            featsR = regions_perImage.at(1)->GetRegionsPositions();

        std::vector<IndMatch> vec_PutativeMatches;
        //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio  ɸѡ
        {
            // Find corresponding points
            matching::DistanceRatioMatch(
                0.8, matching::BRUTE_FORCE_L2,   //matching::BRUTE_FORCE_L2��ʾʹ�ñ���ƥ���㷨
                *regions_perImage.at(0).get(),
                *regions_perImage.at(1).get(),   //��ʾ����ͼ�����������������
                vec_PutativeMatches);   //��ʾ����ĳ���ƥ������

            IndMatchDecorator<float> matchDeduplicator(
                vec_PutativeMatches, featsL, featsR);
            matchDeduplicator.getDeduplicated(vec_PutativeMatches);  //���ڴ�ƥ������ȥ���ظ���ƥ��㡣��������������ǽ� vec_PutativeMatches �����е�ƥ���ȥ�أ������� vec_PutativeMatches �԰������ظ���ƥ���

            std::cout
                << regions_perImage.at(0)->RegionCount() << " #Features on image A" << std::endl
                << regions_perImage.at(1)->RegionCount() << " #Features on image B" << std::endl
                << vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;
        }
        maxx = vec_PutativeMatches.size();
    }



    while (true)
    {


        std::stringstream ss;
        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber1 << ".jpg";
        const std::string jpg_filenameL = ss.str();
        ss.str(""); ss.clear();
        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
        const std::string jpg_filenameR = ss.str();
        // ������ʹ�� jpg_filenameL ����
        ss.str(""); ss.clear();


        Image<unsigned char> imageL, imageR;   //��ͼƬ
        ReadImage(jpg_filenameL.c_str(), &imageL);
        ReadImage(jpg_filenameR.c_str(), &imageR);

        //--
        // Detect regions thanks to an image_describer��������򣬸���ͼ��������
        //--
        using namespace openMVG::features;
        std::unique_ptr<Image_describer> image_describer(new SIFT_Anatomy_Image_describer); //��������������
        std::map<IndexT, std::unique_ptr<features::Regions>> regions_perImage;     //������һ��std::map�������ڴ洢ÿ��ͼ�����������������
                                            //����������������һ�����ڱ�ʾͼ��ֲ������ķ�������ͨ������������ɣ������������������������
        image_describer->Describe(imageL, regions_perImage[0]);   //��imageLͼ�����������ȡ���������������㣬��������洢��regions_perImage[0]��
        image_describer->Describe(imageR, regions_perImage[1]);    //�������������ȡ��

        const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
        const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());//��ͼ��0��ͼ��1����������������ת��ΪSIFT_Regions���ͣ����ֱ�洢��regionsL��regionsR��

        const PointFeatures
            featsL = regions_perImage.at(0)->GetRegionsPositions(),  //����λ����ȡ��������Ϊ���������������
            featsR = regions_perImage.at(1)->GetRegionsPositions();





        // Show both images side by side
        {
            Image<unsigned char> concat;
            ConcatH(imageL, imageR, concat);
            std::string out_filename = "01_concat.jpg";
            WriteImage(out_filename.c_str(), concat);
        }

        //- Draw features on the two image (side by side)
        {
            Features2SVG
            (
                jpg_filenameL,
                { imageL.Width(), imageL.Height() },
                regionsL->Features(),
                jpg_filenameR,
                { imageR.Width(), imageR.Height() },
                regionsR->Features(),
                "02_features.svg"
            );
        }

        std::vector<IndMatch> vec_PutativeMatches;
        //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio  ɸѡ
        {
            // Find corresponding points
            matching::DistanceRatioMatch(
                0.8, matching::BRUTE_FORCE_L2,   //matching::BRUTE_FORCE_L2��ʾʹ�ñ���ƥ���㷨
                *regions_perImage.at(0).get(),
                *regions_perImage.at(1).get(),   //��ʾ����ͼ�����������������
                vec_PutativeMatches);   //��ʾ����ĳ���ƥ������

            IndMatchDecorator<float> matchDeduplicator(
                vec_PutativeMatches, featsL, featsR);
            matchDeduplicator.getDeduplicated(vec_PutativeMatches);  //���ڴ�ƥ������ȥ���ظ���ƥ��㡣��������������ǽ� vec_PutativeMatches �����е�ƥ���ȥ�أ������� vec_PutativeMatches �԰������ظ���ƥ���

            std::cout
                << regions_perImage.at(0)->RegionCount() << " #Features on image A" << frameNumber1 << std::endl
                << regions_perImage.at(1)->RegionCount() << " #Features on image B" << (frameNumber2 + frameNumber1) << std::endl
                << vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;

            // Draw correspondences after Nearest Neighbor ratio filter
            const bool bVertical = true;
            Matches2SVG
            (
                jpg_filenameL,
                { imageL.Width(), imageL.Height() },
                regionsL->GetRegionsPositions(),
                jpg_filenameR,
                { imageR.Width(), imageR.Height() },
                regionsR->GetRegionsPositions(),
                vec_PutativeMatches,
                "03_Matches.svg",
                bVertical
            );
        }

        // Essential geometry filtering of putative matches
        {
            Mat3 K;
            //read K from file
            if (!readIntrinsic(stlplus::create_filespec(outputPath, "K", "txt"), K))
            {
                std::cerr << "Cannot read intrinsic parameters." << std::endl;
                return EXIT_FAILURE;
            }

            const Pinhole_Intrinsic   //��������ڲε�
                camL(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)),
                camR(imageR.Width(), imageR.Height(), K(0, 0), K(0, 2), K(1, 2));

            //A. prepare the corresponding putatives points׼��ƥ��㣬ת���ɾ���
            Mat xL(2, vec_PutativeMatches.size());  //������һ����СΪ2xN������N��PutativeMatches���������ľ���xL
            Mat xR(2, vec_PutativeMatches.size());
            for (size_t k = 0; k < vec_PutativeMatches.size(); ++k) {
                const PointFeature& imaL = featsL[vec_PutativeMatches[k].i_];   //�浽xL��xR�У�Ŀ���ǽ��������Ӧ����������ת��Ϊ������ʽ
                const PointFeature& imaR = featsR[vec_PutativeMatches[k].j_];
                xL.col(k) = imaL.coords().cast<double>();
                xR.col(k) = imaR.coords().cast<double>();
            }
            if (frameNumber2 <= interval && frameNumber1 > 5)
            {
                frameNumber1++;
                frameNumber2 = interval;
                //threads.emplace_back(processFrame, frame.clone(), (frameNumber1), outputPath);
                processFrame(frame.clone(), frameNumber1, outputPath);
                while (frameNumber2--)
                {
                    cap >> frame; // ��ȡ֡
                }
                //threads.emplace_back(processFrame, frame.clone(), (frameNumber1+ interval), outputPath);
                processFrame(frame.clone(), (frameNumber1 + interval), outputPath);

                modo1 = 3;
                frameNumber2 = interval;

            }
            //B. Compute the relative pose thanks to a essential matrix estimation�������λ��  ͨ������������Ƽ���������ƣ��õ��ڵ�
            const std::pair<size_t, size_t> size_imaL(imageL.Width(), imageL.Height());//��������������ͼ��ĳߴ�
            const std::pair<size_t, size_t> size_imaR(imageR.Width(), imageR.Height());//��Щֵ�ڼ������λ��ʱ�ǳ���Ҫ����Ϊ��Ҫ֪������ڲκ�ͼ��ߴ����������λ��
            sfm::RelativePose_Info relativePose_info;  //���ڴ洢������������̬��Ϣ���ڵ���Ϣ
            if (modo1 == 1)
            {

                if (vec_PutativeMatches.size() >= ((maxx / 2)))
                    //   if (sfm::robustRelativePose(&camL, &camR, xL, xR, relativePose_info, size_imaL, size_imaR, 256))
                           /************************************************************************************/
                           //������ʹ���� RANSAC �㷨��³���ع��������̬�����������ڵ㣨inliers������������ͶӰ���
                {
                    std::cout << maxx << std::endl;
                    std::cerr << "+1."  //³�������̬���Ƴɹ�
                        << std::endl;
                    std::stringstream ss;
                    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
                    fs::remove(ss.str());
                    std::cout << "Image file for frame " << (frameNumber2 + frameNumber1) << " has been deleted from " << outputPath << std::endl;
                    frameNumber2 = frameNumber2 + interval;
                }
                else
                {
                    if (vec_PutativeMatches.size() < ((maxx / 2)) && vec_PutativeMatches.size() > (maxx /3))
                        // if (!sfm::robustRelativePose(&camL, &camR, xL, xR, relativePose_info, size_imaL, size_imaR, 256))
                             /************************************************************************************/
                             //������ʹ���� RANSAC �㷨��³���ع��������̬�����������ڵ㣨inliers������������ͶӰ���
                    {
                        std::cerr << " /!\\ Robust relative pose estimation failure."  //³�������̬����ʧ��
                            << std::endl;

                        // Only process frames with interval  //�ٴ�����֮ǰ��ͼƬ

                            // Start a new thread to process the current frame
                          //threads.emplace_back(processFrame, frame.clone(), (frameNumber2 + frameNumber1 - interval), outputPath);
                        processFrame(frame.clone(), (frameNumber2 + frameNumber1 - interval), outputPath);
                        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
                        fs::remove(ss.str());

                        frameNumber1 = (frameNumber2 + frameNumber1) - interval;
                        modo1 = 0;

                        {
                            //threads.emplace_back(processFrame, frame.clone(), (frameNumber1 + interval), outputPath);
                            processFrame(frame.clone(), (frameNumber1 + interval), outputPath);
                            std::stringstream ss;
                            ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber1 << ".jpg";
                            const std::string jpg_filenameL = ss.str();
                            ss.str(""); ss.clear();
                            ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (interval + frameNumber1) << ".jpg";
                            const std::string jpg_filenameR = ss.str();
                            // ������ʹ�� jpg_filenameL ����
                            ss.str(""); ss.clear();


                            Image<unsigned char> imageL, imageR;   //��ͼƬ
                            ReadImage(jpg_filenameL.c_str(), &imageL);
                            ReadImage(jpg_filenameR.c_str(), &imageR);

                            //--
                            // Detect regions thanks to an image_describer��������򣬸���ͼ��������
                            //--
                            using namespace openMVG::features;
                            std::unique_ptr<Image_describer> image_describer(new SIFT_Anatomy_Image_describer); //��������������
                            std::map<IndexT, std::unique_ptr<features::Regions>> regions_perImage;     //������һ��std::map�������ڴ洢ÿ��ͼ�����������������
                                                                //����������������һ�����ڱ�ʾͼ��ֲ������ķ�������ͨ������������ɣ������������������������
                            image_describer->Describe(imageL, regions_perImage[0]);   //��imageLͼ�����������ȡ���������������㣬��������洢��regions_perImage[0]��
                            image_describer->Describe(imageR, regions_perImage[1]);    //�������������ȡ��

                            const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
                            const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());//��ͼ��0��ͼ��1����������������ת��ΪSIFT_Regions���ͣ����ֱ�洢��regionsL��regionsR��

                            const PointFeatures
                                featsL = regions_perImage.at(0)->GetRegionsPositions(),  //����λ����ȡ��������Ϊ���������������
                                featsR = regions_perImage.at(1)->GetRegionsPositions();

                            std::vector<IndMatch> vec_PutativeMatches;
                            //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio  ɸѡ
                            {
                                // Find corresponding points
                                matching::DistanceRatioMatch(
                                    0.8, matching::BRUTE_FORCE_L2,   //matching::BRUTE_FORCE_L2��ʾʹ�ñ���ƥ���㷨
                                    *regions_perImage.at(0).get(),
                                    *regions_perImage.at(1).get(),   //��ʾ����ͼ�����������������
                                    vec_PutativeMatches);   //��ʾ����ĳ���ƥ������

                                IndMatchDecorator<float> matchDeduplicator(
                                    vec_PutativeMatches, featsL, featsR);
                                matchDeduplicator.getDeduplicated(vec_PutativeMatches);  //���ڴ�ƥ������ȥ���ظ���ƥ��㡣��������������ǽ� vec_PutativeMatches �����е�ƥ���ȥ�أ������� vec_PutativeMatches �԰������ظ���ƥ���
                                std::cout
                                    << regions_perImage.at(0)->RegionCount() << " #Features on image A" << frameNumber1 << std::endl
                                    << regions_perImage.at(1)->RegionCount() << " #Features on image B" << (frameNumber1 + interval) << std::endl
                                    << vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;
                            }
                            maxx = vec_PutativeMatches.size();
                            ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (interval + frameNumber1) << ".jpg";
                            fs::remove(ss.str());
                            if (maxx < 10)
                            {
                                {
                                    std::cout
                                        << regions_perImage.at(0)->RegionCount() << " maxx < 10" << frameNumber1 << std::endl
                                        << regions_perImage.at(0)->RegionCount() << " maxx < 10" << frameNumber1 << std::endl
                                        << regions_perImage.at(0)->RegionCount() << " maxx < 10" << frameNumber1 << std::endl
                                        << regions_perImage.at(0)->RegionCount() << " maxx < 10" << frameNumber1 << std::endl;
                                    //threads.emplace_back(processFrame, frame.clone(), (frameNumber1 + interval + interval), outputPath);
                                    processFrame(frame.clone(), (frameNumber1 + interval + interval), outputPath);
                                    std::stringstream ss;
                                    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameNumber1 << ".jpg";
                                    const std::string jpg_filenameL = ss.str();
                                    ss.str(""); ss.clear();
                                    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (interval + frameNumber1 + interval) << ".jpg";
                                    const std::string jpg_filenameR = ss.str();
                                    // ������ʹ�� jpg_filenameL ����
                                    ss.str(""); ss.clear();


                                    Image<unsigned char> imageL, imageR;   //��ͼƬ
                                    ReadImage(jpg_filenameL.c_str(), &imageL);
                                    ReadImage(jpg_filenameR.c_str(), &imageR);

                                    //--
                                    // Detect regions thanks to an image_describer��������򣬸���ͼ��������
                                    //--
                                    using namespace openMVG::features;
                                    std::unique_ptr<Image_describer> image_describer(new SIFT_Anatomy_Image_describer); //��������������
                                    std::map<IndexT, std::unique_ptr<features::Regions>> regions_perImage;     //������һ��std::map�������ڴ洢ÿ��ͼ�����������������
                                                                        //����������������һ�����ڱ�ʾͼ��ֲ������ķ�������ͨ������������ɣ������������������������
                                    image_describer->Describe(imageL, regions_perImage[0]);   //��imageLͼ�����������ȡ���������������㣬��������洢��regions_perImage[0]��
                                    image_describer->Describe(imageR, regions_perImage[1]);    //�������������ȡ��

                                    const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
                                    const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());//��ͼ��0��ͼ��1����������������ת��ΪSIFT_Regions���ͣ����ֱ�洢��regionsL��regionsR��

                                    const PointFeatures
                                        featsL = regions_perImage.at(0)->GetRegionsPositions(),  //����λ����ȡ��������Ϊ���������������
                                        featsR = regions_perImage.at(1)->GetRegionsPositions();

                                    std::vector<IndMatch> vec_PutativeMatches;
                                    //-- Perform matching -> find Nearest neighbor, filtered with Distance ratio  ɸѡ
                                    {
                                        // Find corresponding points
                                        matching::DistanceRatioMatch(
                                            0.8, matching::BRUTE_FORCE_L2,   //matching::BRUTE_FORCE_L2��ʾʹ�ñ���ƥ���㷨
                                            *regions_perImage.at(0).get(),
                                            *regions_perImage.at(1).get(),   //��ʾ����ͼ�����������������
                                            vec_PutativeMatches);   //��ʾ����ĳ���ƥ������

                                        IndMatchDecorator<float> matchDeduplicator(
                                            vec_PutativeMatches, featsL, featsR);
                                        matchDeduplicator.getDeduplicated(vec_PutativeMatches);  //���ڴ�ƥ������ȥ���ظ���ƥ��㡣��������������ǽ� vec_PutativeMatches �����е�ƥ���ȥ�أ������� vec_PutativeMatches �԰������ظ���ƥ���
                                        std::cout
                                            << regions_perImage.at(0)->RegionCount() << " #Features on image A" << frameNumber1 << std::endl
                                            << regions_perImage.at(1)->RegionCount() << " #Features on image B" << (frameNumber1 + interval + interval) << std::endl
                                            << vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;
                                    }
                                    maxx = vec_PutativeMatches.size();
                                    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (interval + frameNumber1 + interval) << ".jpg";
                                    fs::remove(ss.str());
                                }
                            }
                        }

                    }
                    else
                    {
                        if (vec_PutativeMatches.size() < (maxx / 3))
                        {
                            std::cerr << " NO11 <1/3!"  //³�������̬����ʧ��
                                << std::endl;
                            ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
                            fs::remove(ss.str());
                            frameNumber2 = frameNumber2 / 2;
                            modo1 = 2;
                        }


                    }
                }
            }
            else
            {
                if (vec_PutativeMatches.size() >= (maxx / 3))
                    //if (sfm::robustRelativePose(&camL, &camR, xL, xR, relativePose_info, size_imaL, size_imaR, 256))
                        /************************************************************************************/
                        //������ʹ���� RANSAC �㷨��³���ع��������̬�����������ڵ㣨inliers������������ͶӰ���
                {
                    std::cerr << "YES!"  //³�������̬���Ƴɹ�
                        << std::endl;
                    std::stringstream ss;
                    ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
                    fs::remove(ss.str());
                    std::cout << "Image file for frame " << (frameNumber2 + frameNumber1) << " has been deleted from " << outputPath << std::endl;
                    frameNumber2 = frameNumber2 + interval;
                    modo1 = 1;
                }
                else
                {
                    if (vec_PutativeMatches.size() < (maxx / 3))
                        //if (!sfm::robustRelativePose(&camL, &camR, xL, xR, relativePose_info, size_imaL, size_imaR, 256))
                            /************************************************************************************/
                            //������ʹ���� RANSAC �㷨��³���ع��������̬�����������ڵ㣨inliers������������ͶӰ���
                    {
                        std::cerr << " NO!"  //³�������̬����ʧ��
                            << std::endl;
                        ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << (frameNumber2 + frameNumber1) << ".jpg";
                        fs::remove(ss.str());
                        frameNumber2 = frameNumber2 / 2;
                        modo1 = 2;
                    }
                }

            }


            if (modo1 == 1)
            {
                numm11 = interval;
                while (numm11--)
                {
                    if (numm11 == 1)
                    {
                        cap >> frame; // ��ȡ֡
                    }
                    else {
                        cap.grab(); // ����֡
                    }
                }
                //threads.emplace_back(processFrame, frame.clone(), frameNumber2 + frameNumber1, outputPath);
                processFrame(frame.clone(), frameNumber2 + frameNumber1, outputPath);
            }
            else
            {
                if (modo1 == 0)
                {
                    numm11 = frameNumber2;
                    while (numm11--)
                    {
                        if (numm11 == 1)
                        {
                            cap >> frame; // ��ȡ֡
                        }
                        else
                        {
                            cap.grab(); // ����֡
                        }
                    }
                    //threads.emplace_back(processFrame, frame.clone(), frameNumber2 + frameNumber1, outputPath);
                    processFrame(frame.clone(), frameNumber2 + frameNumber1, outputPath);
                }

                else
                {
                    if (modo1 == 2)
                    {
                        numm11 = frameNumber2;
                        int currentFrameNumber = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
                        int desiredFrameNumber = currentFrameNumber - numm11;

                        if (desiredFrameNumber >= 0)
                        {
                            cap.set(cv::CAP_PROP_POS_FRAMES, desiredFrameNumber); // Set the position to the desired frame
                            cap >> frame;
                            //threads.emplace_back(processFrame, frame.clone(), frameNumber2 + frameNumber1, outputPath);
                            processFrame(frame.clone(), frameNumber2 + frameNumber1, outputPath);
                        }
                    }

                }
            }
            //Read the next frame



            if (frame.empty())
                break;
            // Start a new thread to process the current frame




            std::cout << "\nFound an Essential matrix:\n"
                << "\tprecision: " << relativePose_info.found_residual_precision << " pixels\n"
                << "\t#inliers: " << relativePose_info.vec_inliers.size() << "\n"
                << "\t#matches: " << vec_PutativeMatches.size()
                << std::endl;


            // Show Essential validated point
            const bool bVertical = true;
            InlierMatches2SVG
            (
                jpg_filenameL,
                { imageL.Width(), imageL.Height() },
                regionsL->GetRegionsPositions(),
                jpg_filenameR,
                { imageR.Width(), imageR.Height() },
                regionsR->GetRegionsPositions(),
                vec_PutativeMatches,
                relativePose_info.vec_inliers,
                "04_ACRansacEssential.svg",
                bVertical
            );


            //C. Triangulate and export inliers as a PLY scene
            std::vector<Vec3> vec_3DPoints;

            // Setup camera intrinsic and poses
            const Pinhole_Intrinsic intrinsic0(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2));
            const Pinhole_Intrinsic intrinsic1(imageR.Width(), imageR.Height(), K(0, 0), K(0, 2), K(1, 2));

            const Pose3 pose0 = Pose3(Mat3::Identity(), Vec3::Zero());
            const Pose3 pose1 = relativePose_info.relativePose;


            // Init structure by triangulation of the inlier
            std::vector<double> vec_residuals;
            vec_residuals.reserve(relativePose_info.vec_inliers.size() * 4);
            for (const auto inlier_idx : relativePose_info.vec_inliers) {
                const SIOPointFeature& LL = regionsL->Features()[vec_PutativeMatches[inlier_idx].i_];
                const SIOPointFeature& RR = regionsR->Features()[vec_PutativeMatches[inlier_idx].j_];
                // Point triangulation
                Vec3 X;
                const ETriangulationMethod triangulation_method = ETriangulationMethod::DEFAULT;
                if (Triangulate2View
                (
                    pose0.rotation(), pose0.translation(), intrinsic0(LL.coords().cast<double>()),
                    pose1.rotation(), pose1.translation(), intrinsic1(RR.coords().cast<double>()),
                    X,
                    triangulation_method
                ))
                {
                    const Vec2 residual0 = intrinsic0.residual(pose0(X), LL.coords().cast<double>());
                    const Vec2 residual1 = intrinsic1.residual(pose1(X), RR.coords().cast<double>());
                    vec_residuals.emplace_back(std::abs(residual0(0)));
                    vec_residuals.emplace_back(std::abs(residual0(1)));
                    vec_residuals.emplace_back(std::abs(residual1(0)));
                    vec_residuals.emplace_back(std::abs(residual1(1)));
                    vec_3DPoints.emplace_back(X);
                }
            }


            // Display some statistics of reprojection errors
            float dMin, dMax, dMean, dMedian;
            minMaxMeanMedian<float>(vec_residuals.cbegin(), vec_residuals.cend(),
                dMin, dMax, dMean, dMedian);

            std::cout << std::endl
                << "Triangulation residuals statistics:" << "\n"
                << "\t-- Residual min:\t" << dMin << "\n"
                << "\t-- Residual median:\t" << dMedian << "\n"
                << "\t-- Residual max:\t " << dMax << "\n"
                << "\t-- Residual mean:\t " << dMean << std::endl;

        }
    }
    // Join all the threads to wait for them to finish
    //for (auto& thread : threads)
        //thread.join();

    // Release the video capture
    cap.release();

    return EXIT_SUCCESS;
}


bool readIntrinsic(const std::string& fileName, Mat3& K)
{
    // Load the K matrix
    std::ifstream in;
    in.open(fileName.c_str(), std::ifstream::in);
    if (in) {
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                in >> K(j, i);
    }
    else {
        std::cerr << std::endl
            << "Invalid input K.txt file" << std::endl;
        return false;
    }
    return true;
}

/// Export 3D point vector and camera position to PLY format
bool exportToPly(const std::vector<Vec3>& vec_points,
    const std::vector<Vec3>& vec_camPos,
    const std::string& sFileName)
{
    std::ofstream outfile;
    outfile.open(sFileName.c_str(), std::ios_base::out);

    outfile << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " << vec_points.size() + vec_camPos.size()
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;

    for (size_t i = 0; i < vec_points.size(); ++i) {
        outfile << vec_points[i].transpose()
            << " 255 255 255" << "\n";
    }

    for (size_t i = 0; i < vec_camPos.size(); ++i) {
        outfile << vec_camPos[i].transpose()
            << " 0 255 0" << "\n";
    }
    outfile.flush();
    const bool bOk = outfile.good();
    outfile.close();
    return bOk;
}
