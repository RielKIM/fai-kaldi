
#include "faidecoder/fai-decoder-wrapper.h"

int main()
{
    int file_size = 58828;
    char buff[file_size];
    FILE *fin;

    fin = fopen("/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/wav/test_01.wav", "rb");
    while (!feof(fin))
    {
        fread(buff, sizeof(char), file_size, fin);
    }

    FaiDecoderWrapper odw;
    odw.init_configure(800, 0.18, true, false, false, 1, 0.8, 0.0, "1:2:3:4:5");
    odw.load_onlinedecoder();

    odw.run_decode_file(buff, file_size, "123456789");
    odw.decode_finalize("123456789");
    printf("%s\n", odw.get_result());
    odw.reset_onlinedecoder();


    odw.get_fst_addr_to_string();
    odw.get_word_syms_addr_to_string();
    odw.get_feature_info_addr_to_string();
    odw.get_am_nnet_addr_to_string();
    char *fst_addr = odw.get_fst_addr_string();
    char *word_syms_addr = odw.get_word_syms_addr_string();
    char *feature_info_addr = odw.get_feature_info_addr_string();
    char *am_nnet_addr = odw.get_am_nnet_addr_string();

    clock_t p1, p2, p3, p4, p5, p6, p7, p8;
    p1 = clock();
    FaiDecoderWrapper odw2;
    p2 = clock();
    odw2.init_configure(800, 0.18, true, false, false, 1, 0.8, 0.0, "1:2:3:4:5");
    p3 = clock();
    odw2.load_onlinedecoder_addr(fst_addr, word_syms_addr, feature_info_addr, am_nnet_addr);
    p4 = clock();

    odw2.run_decode_websocket(buff, file_size, "123456789");
    p5 = clock();
    odw2.decode_finalize_manual("123456789");
    p6 = clock();
    printf("%s\n", odw2.get_result());
    p7 = clock();
    odw2.reset_onlinedecoder();
    p8 = clock();

    printf("p2-p1 : %f\n", (double)(p2-p1));
    printf("p3-p2 : %f\n", (double)(p3-p2));
    printf("p4-p3 : %f\n", (double)(p4-p3));
    printf("p5-p4 : %f\n", (double)(p5-p4));
    printf("p6-p5 : %f\n", (double)(p6-p5));
    printf("p7-p6 : %f\n", (double)(p7-p6));
    printf("p8-p7 : %f\n", (double)(p8-p7));

    fclose(fin);

    return 0;
} // main()

