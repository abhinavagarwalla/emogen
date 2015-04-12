# Emogen Video Debug Gui Tool

find_package(OpenCV COMPONENTS opencv_core opencv_ml opencv_imgproc
  opencv_highgui opencv_objdetect REQUIRED)

add_executable(emogenvideodebug_cli emogenvideodebug_cli.cpp 
        ACapture.cpp VideoCapture.cpp AGui.cpp DebugGui.cpp
        ../detector/EmoDetector.cpp ../detector/FacePreProcessor.cpp 
        ../detector/SVMEmoDetector.cpp ../detector/ANNEmoDetector.cpp
        ../training/Classifier.cpp ../training/SVMClassifier.cpp 
        ../training/ANNClassifier.cpp ../facedetector/FaceDetector.cpp ../gaborbank/GaborBank.cpp
        ../utils/string_utils.cpp ../utils/matrix_io.cpp )
target_link_libraries(emogenvideodebug_cli  ${OpenCV_LIBS} )
INSTALL(TARGETS emogenvideodebug_cli DESTINATION ${ASSETDIR} )

# Emogen Video Gui Tool
add_executable(emogenvideo_cli emogenvideo_cli.cpp 
        ACapture.cpp VideoCapture.cpp AGui.cpp EmogenGui.cpp
        ../detector/EmoDetector.cpp ../detector/FacePreProcessor.cpp 
        ../detector/SVMEmoDetector.cpp ../detector/ANNEmoDetector.cpp
        ../training/Classifier.cpp ../training/SVMClassifier.cpp 
        ../training/ANNClassifier.cpp ../facedetector/FaceDetector.cpp
        ../gaborbank/GaborBank.cpp
        ../utils/string_utils.cpp ../utils/matrix_io.cpp )
target_link_libraries(emogenvideo_cli ${OpenCV_LIBS} )
INSTALL(TARGETS emogenvideo_cli DESTINATION ${ASSETDIR} )

# Emogen Webcam Gui Tool
add_executable(emogengui_cli emogengui_cli.cpp 
        ACapture.cpp  VideoCapture.cpp AGui.cpp EmogenGui.cpp
        ../detector/EmoDetector.cpp ../detector/FacePreProcessor.cpp 
        ../detector/SVMEmoDetector.cpp ../detector/ANNEmoDetector.cpp
        ../training/Classifier.cpp ../training/SVMClassifier.cpp 
        ../training/ANNClassifier.cpp
        ../facedetector/FaceDetector.cpp
        ../gaborbank/GaborBank.cpp
        ../utils/string_utils.cpp ../utils/matrix_io.cpp )
target_link_libraries(emogengui_cli ${OpenCV_LIBS} )
INSTALL(TARGETS emogengui_cli DESTINATION ${ASSETDIR} )

# Gabor Parameter Tuning Gui Tool
add_executable(gaborgui_cli gaborgui_cli.cpp  
        ImageCapture.cpp ACapture.cpp GaborGui.hpp
        ../gaborbank/GaborBank.cpp
        ../utils/string_utils.cpp ../utils/matrix_io.cpp )
target_link_libraries(gaborgui_cli ${OpenCV_LIBS} )
INSTALL(TARGETS gaborgui_cli DESTINATION ${ASSETDIR} )

