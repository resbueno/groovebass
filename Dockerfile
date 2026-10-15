# GrooveBass roda dentro dos containers do Kicksnare: esta imagem parte da imagem do Kicksnare
# (kicksnare:base) e só acrescenta o pacote groovebass. O web do Kicksnare passa a servir também
# /groovebass e o worker alterna entre as duas filas.
#
#   docker build -t kicksnare:base <repo do kicksnare>
#   docker build -t kicksnare:latest <repo do groovebass>
ARG BASE_IMAGE=kicksnare:base
FROM ${BASE_IMAGE}

ENV GROOVEBASS_BASE=/groovebass

WORKDIR /opt/groovebass
COPY . .
# torch, librosa e numpy já vêm da imagem do Kicksnare; entram só torchcrepe e resampy
RUN pip install .

WORKDIR /app
