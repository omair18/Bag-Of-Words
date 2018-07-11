cmake_minimum_required(VERSION 2.8)
project( BOW )
find_package( OpenCV REQUIRED )
add_executable( createDictionary createDictionary.cpp )
target_link_libraries( createDictionary  ${OpenCV_LIBS} )

add_executable( bovw bovw.cpp )
target_link_libraries( bovw  ${OpenCV_LIBS} )
