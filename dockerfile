FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG ENABLE_INTEL_GPU=false
ENV TZ=Asia/Shanghai \
   LANG=C.UTF-8 \
   LC_ALL=C.UTF-8 \
   COLCON_CACHE_DIR=/workspace/.colcon_cache \
   PIP_NO_CACHE_DIR=1 \
   http_proxy=${HTTP_PROXY} \
   https_proxy=${HTTPS_PROXY}

SHELL ["/bin/bash","-c"]

# ------------------------------------------------------------------
# Base locale/timezone settings — rarely change, keep at the top for reuse
# ------------------------------------------------------------------
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ >/etc/timezone \
 && apt-get update \
 && apt-get install -y --no-install-recommends locales \
 && locale-gen en_US.UTF-8 zh_CN.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Core OS tooling & math/vision libs (stable per env.md / README)
# ------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
   bash-completion \
   build-essential \
   ca-certificates \
   curl \
   ffmpeg \
   git \
   gnupg2 \
   libarmadillo-dev \
   libcanberra-gtk3-module \
   libeigen3-dev \
   libfftw3-dev \
   libgl1 \
   libglib2.0-0 \
   libgoogle-glog-dev \
   libgflags-dev \
   libopencv-dev \
   libpcl-dev \
   libusb-1.0-0-dev \
   lsb-release \
   mesa-utils \
   pkg-config \
   python3 \
   python3-dev \
   python3-pip \
   python3-venv \
   software-properties-common \
   sudo \
   tmux \
   udev \
   vim \
   wget \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# ROS 2 repositories & RMOSS stack (infrequently changed)
# ------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
   > /etc/apt/sources.list.d/ros2.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
   gcc-11 \
   g++-11 \
   python3-colcon-common-extensions \
   python3-rosdep \
   python3-rosinstall-generator \
   python3-vcstool \
   python3-empy \
   python3-lark \
   ros-humble-desktop \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# ------------------------------------------------------------------
# Python toolchain & stable ML runtimes (torch stack rarely changes)
# ------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
   --index-url ${PYTORCH_INDEX_URL} \
   torch==2.2.2 \
   torchvision==0.17.2 \
   torchaudio==2.2.2

# ------------------------------------------------------------------
# Frequently tuned Python packages — keep last to minimize cache bust
# ------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
   numpy==2.1.2 \
   opencv-python==4.12.0.88 \
   ultralytics==8.3.163 \
   pandas \
   matplotlib \
   rich \
   pyyaml \
   pillow

# ------------------------------------------------------------------
# GUI & Intel GPU Support (Added for Intel Iris Xe)
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    intel-opencl-icd \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    PyQt5 \
    PyQt5-sip \
    openvino==2024.0.0 \
    openvino-dev==2024.0.0

# RUN mkdir -p /etc/ros/rosdep/sources.list.d \
#  && curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list \
#      -o /etc/ros/rosdep/sources.list.d/20-default.list \
#  && rosdep update



RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
 && echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> /etc/bash.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]