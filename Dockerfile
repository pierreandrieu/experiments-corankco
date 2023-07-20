# Base image
FROM python:3.8

# Add a non-root user
RUN adduser --disabled-password --gecos '' myuser

# Change to non-root user
USER myuser

# Set the working directory to the home of the non-root user
WORKDIR /home/myuser

# Update PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Install apt-utils
USER root
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
USER myuser

# Upgrade pip and setuptools
RUN /usr/local/bin/python -m pip install --upgrade pip setuptools

# Create necessary directories
RUN mkdir experiments-corankco

# Copy requirements and install python dependencies
COPY --chown=myuser:myuser requirements.txt requirements.txt
RUN pip install --user -r requirements.txt


# Install default-jre
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre python3-dev

# Install Cplex
ARG COSDIR=/opt/CPLEX
ARG CPX_PYVERSION=3.8
ADD ./ILOG_COS_20.10_LINUX_X86_64.bin /tmp/installer
COPY install.properties /tmp/install.properties
RUN chmod u+x /tmp/installer

RUN /tmp/installer -f /tmp/install.properties

RUN rm -f /tmp/installer /tmp/install.properties
RUN apt-get remove -y --purge default-jre && apt-get -y --purge autoremove

RUN rm -rf \
   ${COSDIR}/concert \
   ${COSDIR}/cpoptimizer \
   ${COSDIR}/doc \
   ${COSDIR}/opl \
   ${COSDIR}/python \
   ${COSDIR}/Uninstall \
   ${COSDIR}/cplex/bin \
   ${COSDIR}/cplex/examples \
   ${COSDIR}/cplex/include \
   ${COSDIR}/cplex/lib \
   ${COSDIR}/cplex/matlab \
   ${COSDIR}/cplex/readmeUNIX.html


RUN ls -d ${COSDIR}/cplex/python/* | grep -v ${CPX_PYVERSION} | xargs rm -rf

RUN cd ${COSDIR}/cplex/python/${CPX_PYVERSION}/x86-64_linux && \
	python${CPX_PYVERSION} setup.py install

ENV CPX_PYVERSION ${CPX_PYVERSION}
# end of installation of CPLEX
USER myuser

COPY . .


ENTRYPOINT ["python"]
CMD ["./main.py"]

