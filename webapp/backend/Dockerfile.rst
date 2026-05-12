FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential gfortran \
    libhdf5-dev libnetcdf-dev libpng-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Build original RST codebase
COPY codebase/ /opt/rst/codebase/
COPY build/ /opt/rst/build/

ENV RST_ROOT=/opt/rst
ENV SYSTEM=linux
ENV IPATH=${RST_ROOT}/codebase/include
ENV LIBPATH=${RST_ROOT}/codebase/lib/${SYSTEM}
ENV BINPATH=${RST_ROOT}/codebase/bin/${SYSTEM}
ENV PATH=${BINPATH}:${PATH}
ENV LD_LIBRARY_PATH=${LIBPATH}:${LD_LIBRARY_PATH}

RUN mkdir -p ${LIBPATH} ${BINPATH}

# Build RST libraries and binaries (best-effort; some targets may fail)
RUN cd /opt/rst/codebase && \
    find . -maxdepth 5 -name "makefile" | sort | while read mf; do \
        dir=$(dirname "$mf"); \
        make -C "$dir" -f makefile 2>/dev/null || true; \
    done

COPY webapp/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pythonv2/ /pythonv2/
RUN pip install --no-cache-dir numpy scipy

COPY webapp/backend/ .

ENV PYTHONPATH=/pythonv2
ENV BACKEND_TYPE=rst
ENV RST_BINPATH=${BINPATH}

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
