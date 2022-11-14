## Kaldi comple
https://github.com/kaldi-asr/kaldi/blob/master/INSTALL 의 내용을 따릅니다.


# pre install
```console
apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        gfortran \
        ca-certificates \
        patch \
        ffmpeg \
	vim && \
    rm -rf /var/lib/apt/lists/*
```

# compile 방법
1. kaldi code download : https://github.com/kaldi-asr/kaldi.git
2. commit ID : 1a233a11db0b28aa4966a4e271c839c135de5914

                git reset --hard 1a233a11db0b28aa4966a4e271c839c135de5914  
3. tools 폴더로 이동, ./extras/install_mkl.sh 실행
4. ./extras/check_dependencies.sh 실행 후 OK 확인하고 make -j 4 수행
5. src 폴더로 이동, ./configure --shared --use-cuda=no 실행 (gpu 사용하지 않으므로 no옵션)
6. make depend -j 8 수행 후 make -j 8 수행
 

## Onlinedecoder complie
1. kaldi complie 완료 후 git 의 파일들을 덮어 쓴다
2. 각각의 폴더에서 make를 실행한다.


## bin 파일을 생성 하려면
1. src/onlinedecoderbin_sample 하위 파일을 참고
2. make 파일 수정 : BINFILES = online-decoder 추가
3. .cc 파일에 main 함수 추가하여 빌드

```cpp
int main() {
  int file_size = 58828;
  char buff[file_size];
  FILE *fin;
  clock_t start, end1, end2;

  fin = fopen("/home2/wcyang/kaldi/kaldi/src/onlinedecoderbinmem/data/test_sample/wav/SDRW2000000003/2/SDRW2000000003.1.1.297.wav","rb");
  
  while( !feof(fin) ) {
    fread(buff,sizeof(char),file_size,fin);
  }

  FaiDecoderWrapper odw;
  odw.init_configure(800,"","","","",0.18,true,true,false,false,0,1,0.8,0.0,"1:2:3:4:5");
  odw.load_onlinedecoder();
  odw.run_onlinedecoder(buff, file_size, "123456789");
  odw.decode_finalize("123456789");
  
  printf("%s\n", odw.get_result());
  
  fclose(fin);

  return 0;
} // main()
```
