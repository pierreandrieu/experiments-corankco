FROM python:3.8

RUN adduser myuser
USER myuser
WORKDIR /home/myuser
ENV PATH="/home/myuser/.local/bin:${PATH}"
# RUN /usr/local/bin/python -m pip install --upgrade pip
RUN mkdir experiments-corankco
RUN pip3 install --upgrade --user pip
RUN pip install --upgrade setuptools

# copy and install requirements
COPY --chown=myuser:myuser requirements.txt requirements.txt
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# installation of CPLEX 20.10
USER root
RUN apt-get update && apt-get install -y default-jre
RUN apt-get install -y python3-dev

ARG COSDIR=/opt/CPLEX
ARG CPX_PYVERSION=3.8
ADD ./cplex_studio2211.linux_x86_64.bin /tmp/installer
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

