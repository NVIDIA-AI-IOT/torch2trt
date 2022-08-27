FROM nvcr.io/nvidia/pytorch:21.06-py3


RUN pip3 install termcolor

RUN git clone https://github.com/catchorg/Catch2.git && \
    cd Catch2 && \
    cmake -Bbuild -H. -DBUILD_TESTING=OFF && \
    cmake --build build/ --target install 