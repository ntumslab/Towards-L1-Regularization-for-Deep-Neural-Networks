# Use g++ 4.8 to avoid segmentation fault while using work_sharder
# See https://github.com/tensorflow/tensorflow/issues/13308
CXX := g++
PYTHON_BIN_PATH := python

SRCS := $(shell find tensorflow_sparse_conv2d/cc -name '*.cc')
HEADERS := $(shell find tensorflow_sparse_conv2d/cc -name '*.h')

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS := ${TF_CFLAGS} -fPIC -O3 -std=c++11 -march=native -I.
LDFLAGS := -shared ${TF_LDFLAGS}

TARGET_LIB := tensorflow_sparse_conv2d/python/ops/_sparse_conv2d_ops.so

USE_PARALLEL := 0
ifeq ($(USE_PARALLEL), 1)
	CFLAGS += -DSCONV_PARALLEL
endif


.PHONY: op
op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS) $(HEADERS)
	$(CXX) ${CFLAGS} -o $@ $^ ${LDFLAGS}

test: tensorflow_sparse_conv2d/python/ops/sparse_conv2d_ops_test.py tensorflow_sparse_conv2d/python/ops/sparse_conv2d_ops.py $(TARGET_LIB)
	pytest

.PHONY: clean
clean:
	rm -f $(TARGET_LIB)
