# for CPU compute contexts
#FROM microsoft/mmlspark:plus-0.9.9

# for GPU compute contexts
FROM microsoft/mmlspark:plus-gpu-0.9.9

ENV PREVUSER=$USER
USER root

# install AzCopy on Linux
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-linux?toc=%2fazure%2fstorage%2fblobs%2ftoc.json
RUN apt-get update && apt-get install -y apt-transport-https wget rsync git
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg  && \
        mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg  &&\
        sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-ubuntu-xenial-prod xenial main" > /etc/apt/sources.list.d/dotnetdev.list'  && \
        apt-get update  && \
        apt-get install -y  --no-install-recommends && \
        apt-get install -y dotnet-sdk-2.0.2  && \
        wget -O azcopy.tar.gz https://aka.ms/downloadazcopyprlinux  && \
	tar -xf azcopy.tar.gz  && \
	./install.sh


USER $PREVUSER

