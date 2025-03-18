// LPC.cpp : Defines the entry point for the console application.
//
// includes
#include "stdafx.h"
#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <Windows.h>
#include <algorithm>

#pragma comment(lib, "winmm.lib")
const int NUMPTS = 16025 * 3;   // 3 seconds
short int waveIn[NUMPTS];
void PlayRecord();
void writedataTofile(LPSTR lpData,DWORD dwBufferLength);
void StartRecord();
// Constant definitions
#define M_PI 3.14159265358979323846
#define ll long long
#define ld long double
#define p 12	// order of the predictor
#define frame_size 320
#define frame_shift 80	// frame window shifts by 1/4th of the window size, 3/4th of the data is overlapping between each window
#define N_FRAMES 85		// number of frames to calculate ci's for
#define N_SAMPLES 30
#define INF 1e18
#define scale 1000		// normalization multiplication factor
#define MAX_SIZE 100000	
#define N_DIGITS 6
#define N_SAMPLE_FRAMES 5

// KMM defintions
const int K = 32;
double codebook[K][p];	// each code is a 12x1 vector
const double KMM_delta = 0.00001;
const int U = N_DIGITS * N_FRAMES * N_SAMPLES;	// size of the universe
double universe[U][p];
bool classes[K][U];		// classes[k][m] = 1 if uth vector in the universe was assigned to class k;  
double tokhura_w[p] = {1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};

// LPC defintions
double R[p + 1];	// Ri's
double input[MAX_SIZE];
double data[N_FRAMES * frame_size];	// input frame points
double k[p + 1];
double E[p + 1];
double alpha[p + 1][p + 1];
double a[p + 1];
double c[p + 1];	// raw cepstral coeffecients
double w_ham[frame_size];	// hamming window weights
double w_sin[p + 1];
// double C[N_DIGITS][N_SAMPLE_FRAMES][p + 1];	// average ci's for each frame of 20 samples of each vowel 
bool apply_hamming_window = false;	// set this in main

// folder paths
std::string in_folder = "244101002_dataset/English/text/";	// input folder where audio txt files are stored 
std::string Ci_OUTPUT_FOLDER = "Output/Ci/";		// for ci's
std::string roll_no = "244101002";
std::string codebook_out = "Output/codebook.txt";

// HMM definitions
#define M 32
#define N 5
#define T 85
#define N_MODELS 30

ld pi[N];
ld A[N][N];
ld B[N][M];

ld alpha_fwd[T][N] = {0};
ld beta_bwd[T][N] = {0};
ld delta[T][N] = {0};
short psi[T][N] = {0};
int observation[T];
ld xye[T][N][N];
ld gamma[T][N];

void StartRecord()
{
 int sampleRate = 16025;  
 // 'short int' is a 16-bit type; I request 16-bit samples below
                         // for 8-bit capture, you'd use 'unsigned char' or 'BYTE' 8-bit     types
 HWAVEIN      hWaveIn;
 MMRESULT result;
 WAVEFORMATEX pFormat;
 pFormat.wFormatTag=WAVE_FORMAT_PCM;     // simple, uncompressed format
 pFormat.nChannels=1;                    //  1=mono, 2=stereo
 pFormat.nSamplesPerSec=sampleRate;      // 8.0 kHz, 11.025 kHz, 22.05 kHz, and 44.1 kHz
 pFormat.nAvgBytesPerSec=sampleRate*2;   // =  nSamplesPerSec × nBlockAlign
 pFormat.nBlockAlign=2;                  // = (nChannels × wBitsPerSample) / 8
 pFormat.wBitsPerSample=16;              //  16 for high quality, 8 for telephone-grade
 pFormat.cbSize=0;
 // Specify recording parameters
 result = waveInOpen(&hWaveIn, WAVE_MAPPER,&pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);
 WAVEHDR      WaveInHdr;
 // Set up and prepare header for input
 WaveInHdr.lpData = (LPSTR)waveIn;
 WaveInHdr.dwBufferLength = NUMPTS*2;
 WaveInHdr.dwBytesRecorded=0;
 WaveInHdr.dwUser = 0L;
 WaveInHdr.dwFlags = 0L;
 WaveInHdr.dwLoops = 0L;
 waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
 // Insert a wave input buffer
 result = waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
 // Commence sampling input
 result = waveInStart(hWaveIn);
 printf("recording for 3 seconds...\n");
 Sleep(3 * 1000);
 // Wait until finished recording
 waveInClose(hWaveIn);
 PlayRecord();
}

void PlayRecord()
{
 // const int NUMPTS = 16025 * 3;   // 3 seconds
 int sampleRate = 16025;  
 // 'short int' is a 16-bit type; I request 16-bit samples below
    // for 8-bit capture, you'd    use 'unsigned char' or 'BYTE' 8-bit types
 HWAVEIN  hWaveIn;
 WAVEFORMATEX pFormat;
 pFormat.wFormatTag=WAVE_FORMAT_PCM;     // simple, uncompressed format
 pFormat.nChannels=1;                    //  1=mono, 2=stereo
 pFormat.nSamplesPerSec=sampleRate;      // 44100
 pFormat.nAvgBytesPerSec=sampleRate*2;   // = nSamplesPerSec * n.Channels * wBitsPerSample/8
 pFormat.nBlockAlign=2;                  // = n.Channels * wBitsPerSample/8
 pFormat.wBitsPerSample=16;              //  16 for high quality, 8 for telephone-grade
 pFormat.cbSize=0;
 // Specify recording parameters
 waveInOpen(&hWaveIn, WAVE_MAPPER,&pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);
 WAVEHDR      WaveInHdr;
 // Set up and prepare header for input
 WaveInHdr.lpData = (LPSTR)waveIn;
 WaveInHdr.dwBufferLength = NUMPTS*2;
 WaveInHdr.dwBytesRecorded=0;
 WaveInHdr.dwUser = 0L;
 WaveInHdr.dwFlags = 0L;
 WaveInHdr.dwLoops = 0L;
 waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
 HWAVEOUT hWaveOut;
 printf("playing...\n");
 waveOutOpen(&hWaveOut, WAVE_MAPPER, &pFormat, 0, 0, WAVE_FORMAT_DIRECT);
 waveOutWrite(hWaveOut, &WaveInHdr, sizeof(WaveInHdr)); // Playing the data
 Sleep(3 * 1000); //Sleep for as long as there was recorded
 waveInClose(hWaveIn);
 waveOutClose(hWaveOut);
}

struct HMM_model{
	ld A[N][N];
	ld B[N][M];
	ld pi[N];
	ld P_star;

	void init() {
		P_star = -1;
		for (int i =0; i < N; i++) {
			pi[i] = 0;
			for (int j =0; j < M; j++) {
				A[i][j < N -1 ? j : N - 1] = 0;
				B[i][j] = 0;
			} 
		}
	}
};

struct HMM_model models[N_MODELS];

// K Means Code
void load_universe() {
	FILE *in = fopen("Universe.txt", "r");
	if (in == NULL) {
		printf("couldnt open file Universe.csv\n");
		return;
	}
	int size = 0;
	for (int i =0; i < U && !feof(in); i++) {
		for (int j = 0; j < p; j++) {
			fscanf(in, "%lf ", &universe[i][j]);
		}
		size++;
	}
	printf("Size of the universe: %d\n", size);
	assert(size == U);
	fclose(in);
	/*
	for (int i =0; i < M; i++) {
		for (int j = 0; j < p; j++) {
			printf("%lf,", universe[i][j]);
		}
		printf("\n");
	}
	*/
} 

// return euclidean distacne between to vectors 
double dist_euclid(double* pt1, double* pt2, int size) {
	double acc = 0;
	for (int i =0; i < size; i++) {
		acc += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);   	
	}

	return sqrt(acc);
}

double dist_tokhura(double* pt1, double* pt2, int size) {
	double dist = 0;
	for (int i =0; i < size; i++) {
		dist += tokhura_w[i] * (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);	
	}

	return dist;
}

// <distance, vector index> pair
struct dv_pair {
	double dist;
	int c; 
};

// k is the codebook size
void k_means(int k) {
	// initialize codeboook with predecided centroids
	/*
	for (int i =0; i < k; i++) {
		int c = i * M / k;
		for (int j =0; j < p; j++) {
			codebook[i][j] = universe[c][j];	
		}	
	}
	*/
	printf("\n\nK-Means for k = %d\n\n", k);

	double old_distortion = 0;
	for (int m = 0;; m++) {
		printf("Iteration: %d\n", m);
		for (int i =0; i < M; i++) {
			for (int j =0; j < U; j++) classes[i][j] = 0;
		}

		for (int i = 0; i < U; i++) {
			// classfication of ith points is dv[1] with distance = dv[0]
			struct dv_pair dv = {DBL_MAX, 0};
			for (int j =0; j < k; j++) {
				double dist= dist_euclid(universe[i], codebook[j], p);
				if (dist < dv.dist) {
					dv.dist = dist;
					dv.c = j;
				}
			}
			classes[dv.c][i] = 1;
		}
			
		double curr_distortion = 0;
		// compute distortion and simultaneously
		// recompute centroids
		double new_codebook[K][p];
		for (int i=0; i < K; i++) {
			for (int j=0; j < p; j++) new_codebook[i][j] = 0;
		}
		for (int i =0; i < k; i++) {
			int size = 0;
			for (int j=0; j < U; j++) {
				if (classes[i][j]) {
					curr_distortion += dist_tokhura(universe[j], codebook[i], p); 
					
					size++;
					for (int l=0; l < p; l++) {
						new_codebook[i][l] += universe[j][l];
					} 
				}
			}
			for (int l =0; size != 0 && l < p; l++) {
				new_codebook[i][l] /= size;
			}
		}
		curr_distortion /= M;

		printf("%e - %e = %e\n", old_distortion, curr_distortion, abs(old_distortion - curr_distortion)); 
		if (m > 0) {
			if (abs(old_distortion - curr_distortion) <= KMM_delta) return;
		}
		old_distortion = curr_distortion;
		// swap codebooks
		for (int i =0; i < k; i++) {
			for (int j=0; j < p; j++) {
				// old codebook = new codebook 
				codebook[i][j] = new_codebook[i][j]; 	
			}
		}
	} 
}

const double epsilon = 0.03;
void split_codebook(int curr_size) {
	// split each vector in the notbook into new_size
	int j = curr_size;	// position to insert new code in
	for (int i =0; i < curr_size; i++) {	// for every code in current codebook, split it into 2 new codes
		for (int ii =0; ii < p; ii++) {
			codebook[j][ii] = codebook[i][ii] - epsilon;
			codebook[i][ii] = codebook[i][ii] + epsilon;
		}
		++j;
	}
}

void LBG() {
	// initial codebook with only 1 centroid
	// k means will correct it anyway so just initialized to any vector
	for (int i= 0; i < p; i++) codebook[0][i] = universe[0][i];

	for (int k = 1;; k *= 2) {
		k_means(k);
		if (k ==K) break;
		split_codebook(k);
	}
}

void save_codebook() {
	FILE* out = fopen(codebook_out.c_str(), "w");
	if (out == NULL) {
		printf("couldnt create file: %s\n", codebook_out.c_str());
		return;
	}

	for (int i=0; i < M; i++) {
		for (int j = 0; j < p; j++) {
			fprintf(out, "%lf ", codebook[i][j]);
		}
		fprintf(out, "\n");
	}
	fclose(out);
}

// HMM Code
// solution to problem no. 1

ld p_star = -1;
int q_star_T = -1;
int q_star[T] = {0};

void reset_init_model() {
	std::stringstream ss("");
	ss << "Models/" << "_og_A.txt";
	FILE* in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) {
		for (int j =0;j < N; j++) fscanf(in, "%lf ", &A[i][j]);
	}
	fclose(in);
	
	ss.str("");
	ss << "Models/" << "_og_B.txt";
	in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) {
		for (int j =0;j < M; j++) fscanf(in, "%lf ", &B[i][j]);
	}
	fclose(in);

	ss.str("");
	ss << "Models/" << "_og_pi.txt";
	in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) fscanf(in, "%lf ", &pi[i]);
	fclose(in);

	for (int i=0; i < T; i++) {
		for (int j =0; j <N; j++) {
			alpha_fwd[i][j] =0;
			beta_bwd[i][j] =0;
			psi[i][j] = 0;
			gamma[i][j] =0;
			delta[i][j] = 0;
			for (int k=0; k < N; k++) {
				xye[i][j][k] = 0;
			}
		}
	}

	p_star = -1;
	q_star_T = -1;
	memset(q_star, 0, sizeof(q_star));
}

void reset_avg_model(int digit) {
	std::stringstream ss("");
	ss << "Models/" << digit << "_avg_A.txt";
	FILE* in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) {
		for (int j =0;j < N; j++) fscanf(in, "%lf ", &A[i][j]);
	}
	fclose(in);
	
	ss.str("");
	ss << "Models/" << digit << "_avg_B.txt";
	in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) {
		for (int j =0;j < M; j++) fscanf(in, "%lf ", &B[i][j]);
	}
	fclose(in);

	ss.str("");
	ss << "Models/" << digit << "_avg_pi.txt";
	in = fopen(ss.str().c_str(), "r");
	if (in == NULL) fprintf(stdout, "Couldnt open file: %s\n", ss.str().c_str());
	for (int i=0; i <N; i++) fscanf(in, "%lf ", &pi[i]);
	fclose(in);

	for (int i=0; i < T; i++) {
		for (int j =0; j <N; j++) {
			alpha_fwd[i][j] =0;
			beta_bwd[i][j] =0;
			psi[i][j] = 0;
			gamma[i][j] =0;
			delta[i][j] = 0;
			for (int k=0; k < N; k++) {
				xye[i][j][k] = 0;
			}
		}
	}

	p_star = -1;
	q_star_T = -1;
	memset(q_star, 0, sizeof(q_star));
	// printf("loaded Model for digit: %d\n", digit);
}

ld sol1() {
	/* load data for A, B, pi and observation*/
	for (int i=0; i < N; i++) {
		alpha_fwd[0][i] = pi[i] * B[i][observation[0] - 1];	
	}	

	for (int t= 0; t < T - 1; t++) {
		for (int j = 0; j < N; j++) {
			ld ps = 0.0;	// probablity of previous states
			for (int i =0; i < N; i++) {
				ps += alpha_fwd[t][i] * A[i][j];
			}
			alpha_fwd[t + 1][j] = ps * B[j][observation[t + 1] - 1];
		}
	}
	
	/* this is the probablity for the given model*/
	ld P = 0.0;
	for (int i =0; i < N; i++) P += alpha_fwd[T - 1][i];
	// printf("P = %e\n", P);
	return P;
}
// solution to problem no. 2
void sol2() {
	for (int i =0; i < N; i++) {
		delta[0][i] = pi[i] * B[i][observation[0] - 1];
		psi[0][i] = 0;
	}
		
	for (int t = 1; t < T; t++) {
		for (int j = 0; j < N; j++) {
			ld temp = -1.0;
			short int temp2 = -1;
			for (int i =0; i < N; i++) {
				if (temp < delta[t - 1][i] * A[i][j]) {
					temp = delta[t - 1][i] * A[i][j];
					temp2 = i;
				}
			}
			delta[t][j] = temp * B[j][observation[t] - 1];
			psi[t][j] = temp2;
		}
	}

	for (int i=0; i < N; i++) {
		if (q_star_T == -1 || delta[T - 1][q_star_T] < delta[T -1][i]) q_star_T = i;
		if (p_star < delta[T - 1][i]) p_star = delta[T - 1][i];
	}

	q_star[T - 1] = q_star_T;	
	for (int t = T - 2; t >= 0; t--) {
		q_star[t] = psi[t + 1][q_star[t + 1]];
	}

	// printf("P_star = %e\n", p_star);
	// for (int i=0; i < T; i++) printf("%d ", q_star[i] + 1);
	// printf("\n");
}
// solution to problem no.3
void sol3() {
	// fill beta
	for (int i =0; i < N; i++) beta_bwd[T - 1][i] = 1;
	
	for (int t = T -2; t >= 0; t--) {
		for (int i=0; i < N; i++) {
			for (int j=0; j < N; j++) {
				beta_bwd[t][i] += A[i][j] * B[j][observation[t + 1] - 1] * beta_bwd[t + 1][j];
			}
		}
	}

	for (int t=0; t < T; t++) {
		for (int i =0; i < N; i++) {
			ld dr = 0;
			for (int ii =0; ii < N; ii++) {
				dr += alpha_fwd[t][ii] * beta_bwd[t][ii];
			}
			assert(dr != 0);
			gamma[t][i] = alpha_fwd[t][i] * beta_bwd[t][i] / dr;
		}
	}
	
	for (int t =0; t < T -1; t++) {
		ld P = 0;
		for (int i =0; i < N; i++) {
			for (int j =0; j < N; j++) {
				P += alpha_fwd[t][i] * A[i][j] * B[j][observation[t + 1] - 1] * beta_bwd[t + 1][j]; 
			}
		}
		// printf("%e\n", P);
		assert(P != 0);
		for (int i =0; i < N; i++) {
			for (int j =0; j < N; j++) {
				xye[t][i][j] = alpha_fwd[t][i] * A[i][j] * B[j][observation[t + 1] - 1] * beta_bwd[t + 1][j] / P;
			}
		}
	}
	
	// re-estimation
	// pi
	for (int i =0; i < N; i++) pi[i] = gamma[0][i];
	
	// printf("Aij Matrix: \n");
	// aij
	for (int i=0; i < N; i++) {
		for (int j =0; j <N; j++) {
			ld nr = 0;
			ld dr = 0;
			for (int t = 0; t < T - 1; t++) {
				nr += xye[t][i][j];
				dr += gamma[t][i];
			}
			assert(dr != 0);
			A[i][j] = nr / dr;
			// printf("%e ", _A[i][j]);
		}
		// printf("\n");
	}

	// printf("Bjk Matrix: \n");
	// bjk
	for (int j =0; j < N; j++) {
		for (int k =0; k < M; k++) { 
			ld nr = 0;
			ld dr = 0;
			for (int t = 0; t < T; t++) {
				nr += (observation[t]- 1) == k ? gamma[t][j] : 0.0;
				dr += gamma[t][j];
			}
			B[j][k] = nr / dr;
			// printf("%e ", _B[j][k]);
		}
		// printf("\n");
	}
}

// generate observation sequence from the observation file and store it in observation array
void gen_observation_seq(FILE* in) {
	int i =0;	// index of observation
	double ci_vector[p];
	while(i < T && !feof(in)) {
		for (int i= 0; i < p; i++) ci_vector[i] = 0;
		for (int ii=0; ii < p && !feof(in); ii++) {
			fscanf(in, "%lf ", &ci_vector[ii]);
		}

		double min_dist = DBL_MAX;
		int classification = -1;
		for (int ii=0; ii < M; ii++) {
			double curr_dist = dist_tokhura(codebook[ii], ci_vector, p);
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				classification = ii;
			}
		}
		observation[i++] = classification + 1;	// all the observation code has been 1 indexed
	}
	assert(i == T);
}

// LPC Code
void fill(int v, FILE* in) {
	// zero out
	// -----------
	for (int i =0; i < MAX_SIZE; i++) input[i] = 0;
	for (int i =0; i < N_FRAMES * frame_size; i++) data[i] = 0;
	
	// -----------
	char buf[256];
	for (int i=0 ; i < 5; i++) fscanf(in, "%[^\n]\n", buf);	// skip the first 5 lines of the metadata

	for (int i =0; i < MAX_SIZE && !feof(in); i++) {
		fscanf(in, "%lf\n", &input[i]);
		// printf("%lf\n", input[i]);	
	}

	// find peak
	double peak[2] = {-INF, 0};
	for (int i =0; i < MAX_SIZE; i++) {
		if (peak[0] < input[i]) {
			peak[0] = input[i];
			peak[1] = i;	
		}	
	}
	
	// store 5 frames worth of data starting from 2 * frame_shift data before the peak with 5 frames following after
	// we take overlapping frames shiftedd by frame_shift samples
	int offset = (int)peak[1] - (N_FRAMES / 2) * frame_shift;
	if (offset < 0) offset = 0;
	for (int i=0; i < N_FRAMES * frame_size; i++) {
		data[i] = input[i + offset];
	}
}

void levinson_durbin(int offset) {
	for (int k = 0; k <= p; k++) {
		double autocorrln = 0;	// calculate auto correlations with lag k
		for (int i = offset; i - offset < frame_size - k; i++) {
			autocorrln += (data[i] * data[i + k]);
		}	
		R[k] = autocorrln;
	}
	// k[1] = R[1] / R[0];
	E[0] = R[0];
	for (int i = 1; i <= p; i++) {
		double prod_sum = 0;
		for (int j = 1; j < i; j++) {
			prod_sum += alpha[i- 1][j] * R[i - j];		
		}	

		k[i] = (R[i] - prod_sum) / E[i - 1];
		alpha[i][i] = k[i];
		for (int j = 1; j < i; j++) {
			alpha[i][j] = alpha[i - 1][j] - k[i] * alpha[i - 1][i - j];
		}
		E[i] = (1 - (k[i] * k[i])) * E[i - 1];
	}	

	for (int i =1; i <= p; i++) a[i] = alpha[p][i]; 
}

// calculate cepstral coeffecients and save them in global c array
void gen_cepstral_coeff() {
	for (int i =0; i < 12 + 1; i++) c[i] = 0;
	// sigma is the gain term
	// c[0] = log2(sigma * sigma); 
	for (int m = 1; m <= p; m++) {
		c[m] = a[m];
		for(int k = 1; k <= m- 1; k++) {
			c[m] += (k * c[k] * a[m - k]) / m;   
		}
		// printf("%lf * %lf = %lf\n", c[m], w_sin[m], c[m] * w_sin[m]);
	}
	for (int i = 1; i <= p; i++) c[i] *= w_sin[i];	// apply raised sin window 
}

void preprocess() {
	// perform dc shift and normalization
	double acc = 0;
	double max = -INF;

	for (int i =0; i < frame_size * N_FRAMES; i++) acc += data[i]; 
	double mean = acc / frame_size;
	for (int i =0; i < frame_size * N_FRAMES; i++) {
		data[i] -= mean;
		if ((data[i] < 0 ? -data[i] : data[i]) > max) max = (data[i] < 0 ? -data[i] : data[i]);
	}

	for (int i= 0; i < frame_size * N_FRAMES; i++) data[i] = data[i] * scale / max;	// normalize from [-1000, 1000]
}
// save the HMM averaged model for each digit
void save_model(int digit, std::string suff) {
	std::stringstream ss;
	ss.str("");
	ss << digit;
	std::string a_name = "Models/" + ss.str();
	a_name += "_" + suff;
	a_name += "_A.txt";

	std::string b_name = "Models/" + ss.str();
	b_name += "_" + suff;
	b_name += "_B.txt";

	std::string p_name = "Models/" + ss.str();
	p_name += "_" + suff;
	p_name += "_pi.txt";

	FILE* out;

	out = fopen(p_name.c_str(), "w");
	for (int i=0; i < N; i++) fprintf(out, "%lf ", pi[i]);
	fclose(out);

	out = fopen(a_name.c_str(), "w");
	for (int i=0; i < N; i++) {
		for (int j =0; j < N; j++) {
			fprintf(out, "%lf ", A[i][j]);
		}
		fprintf(out, "\n");
	}
	fclose(out);
	
	out = fopen(b_name.c_str(), "w");
	for (int i=0; i < N; i++) {
		for (int j =0; j < M; j++) {
			fprintf(out, "%lf ", B[i][j]);
		}
		fprintf(out, "\n");
	}
	fclose(out);
}

void load_codebook() {
	FILE* cbfile = fopen("Output/codebook.txt", "r");
	if (cbfile == NULL) {
		fprintf(stdout, "couldnt open file: %s\n", cbfile);
		return;
	}
	while (!feof(cbfile)) {
		for (int i=0; i < M; i++) {
			for (int j=0; j < p; j++) {
				fscanf(cbfile, "%lf ", &codebook[i][j]);
			}
		} 
	}
	fclose(cbfile);
}

const int N_CONTACTS = 6;
const char* contact_names[N_CONTACTS] = {"Bob", "Rahul", "Faiz", "Fahad", "Simon", "John"};
const char* contact_phone[N_CONTACTS] = {"898358395835", "93893485935", "97398359539", "89124842129", "9812931238", "89123819238"};

void print_contact(int classification, ld prob) {
	printf("CONTACT FOUND: ");
	printf("%s %s\n", contact_names[classification], contact_phone[classification]);
	printf("With probablity: %e\n", prob);
}

int _tmain(int argc, _TCHAR* argv[])
{
	printf("Contact Book:\n");
	for (int i= 0; i< N_CONTACTS; i++) {
		printf("%s %s\n", contact_names[i], contact_phone[i]);	
	}
	printf("\n\n\n\n");
	std::stringstream ss;
	std::string observ_filepath;
	FILE* observ_file;

	// LPC Code
	// hamming window calculations
	for (int i = 0; i < frame_size; i++) w_ham[i] = 0.54 - 0.46 * cos((M_PI * 2 * i) / (frame_size - 1));
	
	// tapered window calculation
	double Q = p;
	for (int i = 1; i <= Q; i++) w_sin[i] = 1 + (Q / 2)*sin(M_PI * i / Q);
	int user;
	while(1) {
		printf("1: Retrain\n2: Run test files\n3: Run Live capture\n");
		scanf("%d", &user);
		if (user == 1) {
			std::string filename = "";	//RollNo_v_n.txt 

			FILE* in;
			FILE* outc; // for ci's
			FILE* out_univ = fopen("Universe.txt", "w");
			if (out_univ == NULL) {
				printf("Universe.txt not found\n");
				return 1;
			}
			
			// read the files and write out chopped utterances in each 
			// run calculations for each utterance of each vowel 
			for (int i = 0; i < N_DIGITS; i++) { // 10 digits 
				for (int j = 1; j <= N_SAMPLES + 10; j++) {	// 30 samples per digit
					ss.str("");
					ss << i;
					filename = roll_no+ "_E_" + ss.str(); 
					ss.str("");
					ss << j;
					filename += "_" + ss.str();
					std::string in_filename = in_folder + filename + ".txt";
					in = fopen(in_filename.c_str(), "r");
					if (in == NULL) {
						printf("Couldn't open the file: %s\n exiting\n", in_filename.c_str());
						return 1; 
					} else printf("Reading file: %s\n", in_filename.c_str());

					std::string outc_filename = Ci_OUTPUT_FOLDER + filename + "_ci.txt";
					outc = fopen(outc_filename.c_str(), "w");
					if (outc == NULL) {
						printf("Couldn't open the file: %s \n exiting\n", in_filename.c_str());
						return 1; 
					}
					printf("creating file: %s\n", outc_filename.c_str());

					// for file in files
					fill(i, in);
					// preprocess the data; dc shift and normalization
					preprocess();
					
					// do all the processing
					for (int f =0; f < N_FRAMES; f++) {
						levinson_durbin(f * frame_shift);	// calculate the values on this frame
						gen_cepstral_coeff();
						// accumulate ci's for each vowel for each utterance for each frame
						// for (int m = 1; m <= p; m++) C[i][f][m] += c[m]; // we dont care about utterances
						// write out the ci's for current frame(stored in c array) to a file containing ci's for this utterance
						for (int m= 1; m <= p; m++) {
							fprintf(outc, "%lf ", c[m]);
							fprintf(out_univ, "%lf ", c[m]);
						}
						fprintf(outc, "\n");
						fprintf(out_univ, "\n");
					}
					if (in != NULL) fclose(in);
					if (outc != NULL) fclose(outc);
				}
			}
			
			if (out_univ != NULL) fclose(out_univ);	// the universe has been generated
			// generate codebook
			load_universe();	
			LBG();
			save_codebook();
			// train the models
				
			for (int digit = 0; digit < N_DIGITS; digit++) {
				for (int i =0; i < 3; i++) {
					printf("\n\nCurrent iteration for averageing: %d\n", i);
					for (int sample = 1; sample <= N_SAMPLES ; sample++) {
						// generate observation sequence for current sample
						ss.str("");
						ss << digit << "_" << sample << "_ci.txt";
						observ_filepath = Ci_OUTPUT_FOLDER + "244101002_E_" + ss.str();
						if (!(observ_file = fopen(observ_filepath.c_str(), "r"))) {
							printf("Couldnt open file: %s\n EXITING...\n", observ_filepath.c_str());
							return 1;
						} else printf("Reading observations from file: %s\n", observ_filepath.c_str());

						gen_observation_seq(observ_file);
						fclose(observ_file);
						// printf("current observation sequence: \n");
						// for (int i=0; i < T; i++) printf("%d ", observation[i]);
						// printf("\n");

						if (i ==0) {
							reset_init_model();
						} else {
							reset_avg_model(digit);
						}
						for (int ii =0; ii < 100; ii++) {	
							sol1();
							sol2();
							sol3();
						}
						// printf("Final P_star = %e\n", p_star);
						// for (int i=0; i < T; i++) printf("%d ", q_star[i] + 1);
						// printf("\n");
						// save each model
						models[sample - 1].init();
						models[sample - 1].P_star = p_star;
						for (int i =0; i < N; i++) {
							models[sample - 1].pi[i] = pi[i];
							for (int j=0; j < M; j++) {
								models[sample - 1].B[i][j] = B[i][j]; 
							}
							for (int j =0; j < N; j++) {
								models[sample - 1].A[i][j] = A[i][j];
							}
						}
					}
					// 30 models should have been saved now
					// average out the 30 models, this average model is now the starting point for this
					for (int i =0; i < N; i++) {
						for (int j =0; j < M; j++) {
							B[i][j] = 0;
							for (int m =0; m < N_SAMPLES; m++) {
								B[i][j] += models[m].B[i][j];
							}
							B[i][j] /= N_SAMPLES;
						}
					}
					for (int i =0; i < N; i++) {
						for (int j =0; j < N; j++) {
							A[i][j] = 0;
							for (int m =0; m < N_SAMPLES; m++) {
								A[i][j] += models[m].A[i][j];
							}
							A[i][j] /= N_SAMPLES;
						}
					}
					for (int i =0; i < N; i++) {
						pi[i] =0;
						for (int m=0; m < N_SAMPLES; m++) {
							pi[i] += models[m].pi[i];
						}
						pi[i] /= N_SAMPLES;
					}

					// save average model
					save_model(digit, "avg");
				}
				// overwrites to the overall average model
				save_model(digit, "avg");
			}
		} else if (user == 2) {
			// test the code
			load_codebook();	
			double count = 0;
			for (int i = 0; i < N_DIGITS; i++) {
				printf("\n\nCurrent samples index %d:\n", i);
				for (int sample = 31; sample <= 40; sample++) {
					ld max_P = 0.0;
					int classification = -1;
					// generate observation sequence for current sample
					ss.str("");
					ss << i << "_" << sample << "_ci.txt";
					observ_filepath = Ci_OUTPUT_FOLDER + "244101002_E_" + ss.str();
					if (!(observ_file = fopen(observ_filepath.c_str(), "r"))) {
						printf("Couldnt open file: %s\n EXITING...\n", observ_filepath.c_str());
						return 1;
					} else printf("Reading observations from file: %s\n", observ_filepath.c_str());

					gen_observation_seq(observ_file);
					fclose(observ_file);
					// printf("current observation sequence: \n");
					// for (int i=0; i < T; i++) printf("%d ", observation[i]);
					// printf("\n");
					for(int digit = 0; digit < N_DIGITS; digit++) {
						reset_avg_model(digit);
						ld P= sol1();
						if (max_P < P) {
							max_P = P;
							classification = digit;
						}
					}
					printf("Verdict: %d with Probablity: %e\n", classification, max_P);
					count += classification == i;
				}
			} 
			printf("Success ratio: %lf \n", count / (N_DIGITS * 10));
		} else if (user == 3) {
			// Live recording testing 
			FILE* record_out;
			load_codebook();

			int user;
			for (int i=0;; i++) {
				fprintf(stdout, "Enter 0 to record, 1 to exit: ");
				fscanf(stdin, "%d", &user);
				if (user == 1) break;
				StartRecord();
				ss.str("");
				ss << "Recording_" << i << ".txt";
				record_out = fopen(ss.str().c_str(), "w");
				assert(record_out != NULL);
				for (int i=0; i < NUMPTS; i++) {
				   fprintf(record_out, "%d\n", waveIn[i]);
				}
				fclose(record_out);
				FILE* record_in = fopen(ss.str().c_str(), "r");
				assert(record_in != NULL);
				ss.str("");
				ss << "Recording_" << i << "_ci.txt";
				FILE* outc = fopen(ss.str().c_str(), "w");
				FILE* inc;
				assert(outc != NULL);
				
				fill(i, record_in);
				if (record_in != NULL) fclose(record_in);

				preprocess();
				for (int f =0; f < N_FRAMES; f++) {
					levinson_durbin(f * frame_shift);	// calculate the values on this frame
					gen_cepstral_coeff();
					for (int m= 1; m <= p; m++) {
						fprintf(outc, "%lf ", c[m]);
					}
					fprintf(outc, "\n");
				}
				fclose(outc);
				inc = fopen(ss.str().c_str(), "r");
				gen_observation_seq(inc);
				ld max_P = 0.0;
				int classification = -1;
				for(int digit = 0; digit < N_DIGITS; digit++) {
					reset_avg_model(digit);
					ld _P = sol1();
					assert(_P >= 0);
					printf("Model %d with probablity: %e\n", digit, _P);
					if (max_P < _P) {
						max_P = _P;
						classification = digit;
					}
				}
				if (classification != -1) print_contact(classification, max_P);
				else printf("Not recognized\n");
				// printf("Verdict: %d with Probablity: %e\n", classification, max_P);
			}
		} else printf("Try Again\n");
	}
	return 0;
}