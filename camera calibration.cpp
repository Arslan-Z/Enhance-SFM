#include <opencv2/imgproc/types_c.h>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
 
Mat image, img_gray;
int BOARDSIZE[2]{ 6,9 };//���̸�ÿ��ÿ�нǵ����
int main()
{
	vector<vector<Point3f>> objpoints_img;//�������̸��Ͻǵ����ά����
	vector<Point3f> obj_world_pts;//��ά��������
	vector<vector<Point2f>> images_points;//�������нǵ�
	vector<Point2f> img_corner_points;//����ÿ��ͼ��⵽�Ľǵ�
	vector<String> images_path;//����������Ŷ�ȡͼ��·��
 
	string image_path = "/home/titan/Calibration/image/pictures/*.jpg";//������ͼ·��	
	glob(image_path, images_path);//��ȡָ���ļ�����ͼ��
 
	//ת��������ϵ
	for (int i = 0; i < BOARDSIZE[1]; i++)
	{
		for (int j = 0; j < BOARDSIZE[0]; j++)
		{
			obj_world_pts.push_back(Point3f(j, i, 0));
		}
	}
 
	for (int i = 0; i < images_path.size(); i++)
	{
		image = imread(images_path[i]);
		cvtColor(image, img_gray, COLOR_BGR2GRAY);
		//���ǵ�
		bool found_success = findChessboardCorners(img_gray, Size(BOARDSIZE[0], BOARDSIZE[1]),
			img_corner_points,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
 
		//��ʾ�ǵ�
		if (found_success)
		{
			//������ֹ����
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
 
			//��һ����ȡ�����ؽǵ�
			cornerSubPix(img_gray, img_corner_points, Size(11, 11),
				Size(-1, -1), criteria);
 
			//���ƽǵ�
			drawChessboardCorners(image, Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points,
				found_success);
 
			objpoints_img.push_back(obj_world_pts);//����������ϵ���������ϵ
			images_points.push_back(img_corner_points);
		}
		//char *output = "image";
		char text[] = "image";
		char *output = text;
		imshow(output, image);
		waitKey(200);
 
	}
 
	/*
	�����ڲκͻ���ϵ����
	*/
 
	Mat cameraMatrix, distCoeffs, R, T;//�ڲξ��󣬻���ϵ������ת����ƫ����
	calibrateCamera(objpoints_img, images_points, img_gray.size(),
		cameraMatrix, distCoeffs, R, T);
 
	cout << "cameraMatrix:" << endl;
	cout << cameraMatrix << endl;
 
	cout << "*****************************" << endl;
	cout << "distCoeffs:" << endl;
	cout << distCoeffs << endl;
	cout << "*****************************" << endl;
 
	cout << "Rotation vector:" << endl;
	cout << R << endl;
 
	cout << "*****************************" << endl;
	cout << "Translation vector:" << endl;
	cout << T << endl;
 
	///*
	//����ͼ��У׼
	//*/
	Mat src, dst;
	src = imread("/home/titan/Calibration/image/pictures/02.jpg");  //��ȡУ��ǰͼ��
	undistort(src, dst, cameraMatrix, distCoeffs);
 
	char texts[] = "image_dst";
	char *dst_output = texts;
	//char *dst_output = "image_dst";
	imshow(dst_output, dst);
	waitKey(100);
	imwrite("/home/titan/Calibration/image/pictures/002.jpg", dst);  //У����ͼ��
 
	destroyAllWindows();//������ʾ����
	system("pause");
	return 0;
}