
#include <arm_neon.h>
#include "tensor.hpp"

#define vfloat4 float32x4_t

int min(float a, float b)
{
    return (a < b) ? a : b;
}

void naive_matmul_row_major_kernel(int M, int N, int K, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float res = 0;
            for (int n = 0; n < N; n++)
            {
                res += aData[s_a * n + m] * bData[n * s_b + k];
            }
            outData[s_out * m + k] += res;
        }
    }
}

void matmul_4x4_micro_kernel_row_major(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{
    vfloat4 res1 = {0, 0, 0, 0};
    vfloat4 res2 = {0, 0, 0, 0};
    vfloat4 res3 = {0, 0, 0, 0};
    vfloat4 res4 = {0, 0, 0, 0};

    vfloat4 res5 = {0, 0, 0, 0};
    vfloat4 res6 = {0, 0, 0, 0};
    vfloat4 res7 = {0, 0, 0, 0};
    vfloat4 res8 = {0, 0, 0, 0};

    vfloat4 res9 = {0, 0, 0, 0};
    vfloat4 res10 = {0, 0, 0, 0};
    vfloat4 res11 = {0, 0, 0, 0};
    vfloat4 res12 = {0, 0, 0, 0};

    vfloat4 res13 = {0, 0, 0, 0};
    vfloat4 res14 = {0, 0, 0, 0};
    vfloat4 res15 = {0, 0, 0, 0};
    vfloat4 res16 = {0, 0, 0, 0};

    vfloat4 va1, va2, va3, va4, vb1, vb2, vb3, vb4;
    for (int n = 0; n < N - 3; n += 4)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);

        va2 = vld1q_f32(aData + s_a * (n + 1));
        vb2 = vld1q_f32(bData + s_b * (n + 1));
        res5 = vfmaq_laneq_f32(res5, vb2, va2, 0);
        res6 = vfmaq_laneq_f32(res6, vb2, va2, 1);
        res7 = vfmaq_laneq_f32(res7, vb2, va2, 2);
        res8 = vfmaq_laneq_f32(res8, vb2, va2, 3);

        va3 = vld1q_f32(aData + s_a * (n + 2));
        vb3 = vld1q_f32(bData + s_b * (n + 2));
        res9 = vfmaq_laneq_f32(res9, vb3, va3, 0);
        res10 = vfmaq_laneq_f32(res10, vb3, va3, 1);
        res11 = vfmaq_laneq_f32(res11, vb3, va3, 2);
        res12 = vfmaq_laneq_f32(res12, vb3, va3, 3);

        va4 = vld1q_f32(aData + s_a * (n + 3));
        vb4 = vld1q_f32(bData + s_b * (n + 3));
        res13 = vfmaq_laneq_f32(res13, vb4, va4, 0);
        res14 = vfmaq_laneq_f32(res14, vb4, va4, 1);
        res15 = vfmaq_laneq_f32(res15, vb4, va4, 2);
        res16 = vfmaq_laneq_f32(res16, vb4, va4, 3);
    }

    for (int n = N - N % 4; n < N; n++)
    {
        va1 = vld1q_f32(aData + s_a * n);
        vb1 = vld1q_f32(bData + s_b * n);
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb1, va1, 1);
        res3 = vfmaq_laneq_f32(res3, vb1, va1, 2);
        res4 = vfmaq_laneq_f32(res4, vb1, va1, 3);
    }

    res1 = vaddq_f32(res1, res5);
    res5 = vld1q_f32(outData);
    res9 = vaddq_f32(res9, res13);
    res2 = vaddq_f32(res2, res6);
    res6 = vld1q_f32(outData + s_out);
    res10 = vaddq_f32(res10, res14);
    res3 = vaddq_f32(res3, res7);
    res7 = vld1q_f32(outData + 2 * s_out);
    res11 = vaddq_f32(res11, res15);
    res4 = vaddq_f32(res4, res8);
    res8 = vld1q_f32(outData + 2 * s_out);
    res12 = vaddq_f32(res12, res16);
    res1 = vaddq_f32(res1, res5);
    res2 = vaddq_f32(res2, res6);
    res3 = vaddq_f32(res3, res7);
    res4 = vaddq_f32(res4, res8);
    res1 = vaddq_f32(res1, res9);
    res2 = vaddq_f32(res2, res10);
    res3 = vaddq_f32(res3, res11);
    res4 = vaddq_f32(res4, res12);
    
    vst1q_f32(outData, res1);
    vst1q_f32(outData + s_out, res2);
    vst1q_f32(outData + 2 * s_out, res3);
    vst1q_f32(outData + 3 * s_out, res4);
}

void matmul_12x8_micro_kernel_row_major(int N, float *aData, float *bData, float *outData, int s_a, int s_b, int s_out)
{

    vfloat4 res1 = vld1q_f32(outData);
    vfloat4 res2 = vld1q_f32(outData + 4);
    vfloat4 res3 = vld1q_f32(outData + s_out);
    vfloat4 res4 = vld1q_f32(outData + s_out + 4);

    vfloat4 res5 = vld1q_f32(outData + 2 * s_out);
    vfloat4 res6 = vld1q_f32(outData + 2 * s_out + 4);
    vfloat4 res7 = vld1q_f32(outData + 3 * s_out);
    vfloat4 res8 = vld1q_f32(outData + 3 * s_out + 4);

    vfloat4 res9 = vld1q_f32(outData + 4 * s_out);
    vfloat4 res10 = vld1q_f32(outData + 4 * s_out + 4);
    vfloat4 res11 = vld1q_f32(outData + 5 * s_out);
    vfloat4 res12 = vld1q_f32(outData + 5 * s_out + 4);

    vfloat4 res13 = vld1q_f32(outData + 6 * s_out);
    vfloat4 res14 = vld1q_f32(outData + 6 * s_out + 4);
    vfloat4 res15 = vld1q_f32(outData + 7 * s_out);
    vfloat4 res16 = vld1q_f32(outData + 7 * s_out + 4);

    vfloat4 res17 = vld1q_f32(outData + 8 * s_out);
    vfloat4 res18 = vld1q_f32(outData + 8 * s_out + 4);
    vfloat4 res19 = vld1q_f32(outData + 9 * s_out);
    vfloat4 res20 = vld1q_f32(outData + 9 * s_out + 4);

    vfloat4 res21 = vld1q_f32(outData + 10 * s_out);
    vfloat4 res22 = vld1q_f32(outData + 10 * s_out + 4);
    vfloat4 res23 = vld1q_f32(outData + 11 * s_out);
    vfloat4 res24 = vld1q_f32(outData + 11 * s_out + 4);

    vfloat4 va1 = vld1q_f32(aData);
    vfloat4 vb1 = vld1q_f32(bData);
    vfloat4 va2 = vld1q_f32(aData + 4);
    vfloat4 vb2 = vld1q_f32(bData + 4);
    vfloat4 va3 = vld1q_f32(aData + 8);

    aData += s_a;
    bData += s_b;

    for (size_t n = 0; n < N; n++) // n
    {
        res1 = vfmaq_laneq_f32(res1, vb1, va1, 0);
        res2 = vfmaq_laneq_f32(res2, vb2, va1, 0); // Row1

        res3 = vfmaq_laneq_f32(res3, vb1, va1, 1);
        res4 = vfmaq_laneq_f32(res4, vb2, va1, 1); // Row2

        res5 = vfmaq_laneq_f32(res5, vb1, va1, 2);
        res6 = vfmaq_laneq_f32(res6, vb2, va1, 2); // Row3

        res7 = vfmaq_laneq_f32(res7, vb1, va1, 3);
        res8 = vfmaq_laneq_f32(res8, vb2, va1, 3); // Row4
        va1 = vld1q_f32(aData);

        res9 = vfmaq_laneq_f32(res9, vb1, va2, 0);
        res10 = vfmaq_laneq_f32(res10, vb2, va2, 0); // Row8

        res11 = vfmaq_laneq_f32(res11, vb1, va2, 1);
        res12 = vfmaq_laneq_f32(res12, vb2, va2, 1); // Row9

        res13 = vfmaq_laneq_f32(res13, vb1, va2, 2);
        res14 = vfmaq_laneq_f32(res14, vb2, va2, 2); // Row10

        res15 = vfmaq_laneq_f32(res15, vb1, va2, 3);
        res16 = vfmaq_laneq_f32(res16, vb2, va2, 3); // Row11
        va2 = vld1q_f32(aData + 4);

        res17 = vfmaq_laneq_f32(res17, vb1, va3, 0);
        res18 = vfmaq_laneq_f32(res18, vb2, va3, 0); // Row12

        res19 = vfmaq_laneq_f32(res19, vb1, va3, 1);
        res20 = vfmaq_laneq_f32(res20, vb2, va3, 1); // Row13

        res21 = vfmaq_laneq_f32(res21, vb1, va3, 2);
        res22 = vfmaq_laneq_f32(res22, vb2, va3, 2); // Row14

        res23 = vfmaq_laneq_f32(res23, vb1, va3, 3);
        res24 = vfmaq_laneq_f32(res24, vb2, va3, 3); // Row15
        vb1 = vld1q_f32(bData);
        vb2 = vld1q_f32(bData + 4);
        va3 = vld1q_f32(aData + 8);

        aData += s_a;
        bData += s_b;
    }

    vst1q_f32(outData, res1);
    vst1q_f32(outData + 4, res2);
    vst1q_f32(outData + s_out, res3);
    vst1q_f32(outData + s_out + 4, res4);
    vst1q_f32(outData + 2 * s_out, res5);
    vst1q_f32(outData + 2 * s_out + 4, res6);
    vst1q_f32(outData + 3 * s_out, res7);
    vst1q_f32(outData + 3 * s_out + 4, res8);
    vst1q_f32(outData + 4 * s_out, res9);
    vst1q_f32(outData + 4 * s_out + 4, res10);
    vst1q_f32(outData + 5 * s_out, res11);
    vst1q_f32(outData + 5 * s_out + 4, res12);
    vst1q_f32(outData + 6 * s_out, res13);
    vst1q_f32(outData + 6 * s_out + 4, res14);
    vst1q_f32(outData + 7 * s_out, res15);
    vst1q_f32(outData + 7 * s_out + 4, res16);
    vst1q_f32(outData + 8 * s_out, res17);
    vst1q_f32(outData + 8 * s_out + 4, res18);
    vst1q_f32(outData + 9 * s_out, res19);
    vst1q_f32(outData + 9 * s_out + 4, res20);
    vst1q_f32(outData + 10 * s_out, res21);
    vst1q_f32(outData + 10 * s_out + 4, res22);
    vst1q_f32(outData + 11 * s_out, res23);
    vst1q_f32(outData + 11 * s_out + 4, res24);
}

void matmul(int M, int N, int K, float *aData, float *bData, float *outData)
{

    int lda = M;
    int ldb = K;
    int ldout = K;

    auto matmul_12x8 = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 11; m += 12)
        {
            for (int k = 0; k < K - 7; k += 8)
            {
                matmul_12x8_micro_kernel_row_major(N, aData + m, bData + k, outData + k + ldo * m, lda, ldb, ldo);
            }
        }
    };

    auto matmul_4x4 = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        for (int m = 0; m < M - 3; m += 4)
        {
            for (int k = 0; k < K - 3; k += 4)
            {
                matmul_4x4_micro_kernel_row_major(N, aData + m, bData + k, outData + k + ldo * m, lda, ldb, ldo);
            }
        }
    };

    auto naive_matmul = [&](int M, int N, int K, float *aData, int lda, float *bData, int ldb, float *outData, int ldo)
    {
        naive_matmul_row_major_kernel(M, N, K, aData, bData, outData, lda, ldb, ldo);
    };

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_12x8(M, n_, K, aData + (M * n), lda, bData + (K * n), ldb, outData, ldout);
    }

    int mleft = M;
    int kleft = K;
    if (M / 12 > 0 && K / 8 > 0)
    {
        mleft = M % 12;
        kleft = K % 8;
    }
    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_4x4(M, n_, kleft, aData + (M * n), lda, bData + (K * n) + (K - kleft), ldb, outData + K-kleft, ldout);
    }

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        matmul_4x4(mleft, n_, K - kleft, aData + (M * n) + (M - mleft), lda, bData + (K * n), ldb, outData + ldout * (M - mleft), ldout);
    }

    if (M / 4 > 0 && K / 4 > 0)
    {
        mleft = M % 4;
        kleft = K % 4;
    }

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        naive_matmul(M, n_, kleft, aData + (M * n), lda, bData + (K * n) + (K - kleft), ldb, outData + (K - kleft), ldout);
    }

    for (int n = 0; n < N; n += 128)
    {
        int n_ = min(N - n, 128);
        naive_matmul(mleft, n_, K - kleft, aData + (M * n) + (M - mleft), lda, bData + (K * n), ldb, outData + ldout * (M - mleft), ldout);
    }
}