/*
 * Empty C++ Application
 */

/*
 * Empty C++ Application
 */
#include "datate.h"

using namespace std;

#define NN_DEVICE_ID XPAR_MYPROJECT_0_DEVICE_ID
#define UART_DEVICE_ID XPAR_XUARTPS_0_DEVICE_ID


static XMyproject z;



#define N_INPUTS 14248
//14248


ap_uint<32>** database_builder()
{
	long long unsigned int k = 0;

	//fixed_point<short,10> fip;
	ap_fixed<16,6> fip;
	float fldata;
	//fixed_point<short,10> fixdata[N_INPUTS][27];
	ap_fixed<16,6> fixdata[N_INPUTS][27];
	long long preview;



	for(int j = 0; j < N_INPUTS; ++j)
	{
		for (int i = 0; i < 27; ++i)
			{
				fldata = data[k];
				//preview = fldata*1024;
				//xil_printf("Iteration%d\n",k);
				fip = fldata;
				fixdata[j][i] = fip;
				++k;
			}
	}
	//file.close();

	ap_uint<32>** worddata = new ap_uint<32>*[N_INPUTS];
	for (int i = 0; i < N_INPUTS; ++i)
	{
		worddata[i] = new ap_uint<32>[14];
	}
	k = 0;

	for(int j = 0; j < N_INPUTS; ++j)
	{
		//clog << "wordata " << j << '\t';
		for (int i = 0; i < 26; i+=2)
		{
			worddata[j][k] = ((ap_uint<16>)fixdata[j][i+1].range(),(ap_uint<16>)fixdata[j][i].range());
			//worddata[j][k] = fixdata[j][i+1] + fixdata[j][i];
			++k;
		}
		fip = 0;
		worddata[j][13] = ((ap_uint<16>)fixdata[j][26].range(),(ap_uint<16>)fip.range());
		//worddata[j][13] = fixdata[j][26]+fip;
		k = 0;
	}
	return worddata;

}

XMyproject_Q_dense_10_input_v input_builder(XMyproject_Q_dense_10_input_v &datatest, ap_uint<32>* ar)
{
	datatest.word_0 = ar[0].to_uint();
	datatest.word_1 = ar[1].to_uint();
	datatest.word_2 = ar[2].to_uint();
	datatest.word_3 = ar[3].to_uint();
	datatest.word_4 = ar[4].to_uint();
	datatest.word_5 = ar[5].to_uint();
	datatest.word_6 = ar[6].to_uint();
	datatest.word_7 = ar[7].to_uint();
	datatest.word_8 = ar[8].to_uint();
	datatest.word_9 = ar[9].to_uint();
	datatest.word_10 = ar[10].to_uint();
	datatest.word_11= ar[11].to_uint();
	datatest.word_12 = ar[12].to_uint();
	datatest.word_13 = ar[13].to_uint();
	return datatest;
}

u32* input_reader(XMyproject_Q_dense_10_input_v datatest)
{
	unsigned int* ar = new unsigned int[28];
	ar[0] = (datatest.word_0 & 0xFFFF0000) >> 16;
	ar[1] = (datatest.word_0 & 0xFFFF);
	ar[2] = (datatest.word_1 & 0xFFFF0000) >> 16;
	ar[3] = (datatest.word_1 & 0xFFFF);
	ar[4] = (datatest.word_2 & 0xFFFF0000) >> 16;
	ar[5] = (datatest.word_2 & 0xFFFF);
	ar[6] = (datatest.word_3 & 0xFFFF0000) >> 16;
	ar[7] = (datatest.word_3 & 0xFFFF);
	ar[8] = (datatest.word_4 & 0xFFFF0000) >> 16;
	ar[9] = (datatest.word_4 & 0xFFFF);
	ar[10] = (datatest.word_5 & 0xFFFF0000) >> 16;
	ar[11] = (datatest.word_5 & 0xFFFF);
	ar[12] = (datatest.word_6 & 0xFFFF0000) >> 16;
	ar[13] = (datatest.word_6 & 0xFFFF);
	ar[14] = (datatest.word_7 & 0xFFFF0000) >> 16;
	ar[15] = (datatest.word_7 & 0xFFFF);
	ar[16] = (datatest.word_8 & 0xFFFF0000) >> 16;
	ar[17] = (datatest.word_8 & 0xFFFF);
	ar[18] = (datatest.word_9 & 0xFFFF0000) >> 16;
	ar[19] = (datatest.word_9 & 0xFFFF);
	ar[20] = (datatest.word_10 & 0xFFFF0000) >> 16;
	ar[21] = (datatest.word_10 & 0xFFFF);
	ar[22] = (datatest.word_11 & 0xFFFF0000) >> 16;
	ar[23] = (datatest.word_11 & 0xFFFF);
	ar[24] = (datatest.word_12 & 0xFFFF0000) >> 16;
	ar[25] = (datatest.word_12 & 0xFFFF);
	ar[26] = (datatest.word_13 & 0xFFFF0000) >> 16; //It should be always zero!
	ar[27] = (datatest.word_13 & 0xFFFF);
	return ar;

}

void hw_inference(ap_uint<32>** worddata,XMyproject* z)
{
	XMyproject_Q_dense_10_input_v datatest;
	u32 res_hw[N_INPUTS];
	unsigned int i = 0;
	fixed_point<short,10> ti;
	ap_fixed<32,20> tiTot;
	XTime tStart, tEnd,tStartTot,tEndTot;
	XTime_StartTimer();
	XTime_GetTime(&tStartTot);
	while(i < N_INPUTS)
	{
		input_builder(datatest,worddata[i]);

		XMyproject_Set_q_dense_10_input_V(z,datatest);
		//xil_printf("\nFirst written word = 0x%X \n", datatest.word_0);

		XMyproject_Start(z);
		XMyproject_Q_dense_10_input_v intest = XMyproject_Get_q_dense_10_input_V(z);
		u32* pattern = input_reader(intest);

		unsigned int asd = 0;
		XTime_GetTime(&tStart);

		do {
			res_hw[i] = XMyproject_Get_layer19_out_0_V(z);
			++asd;
			if (asd >= 50) break;


			} while (!XMyproject_IsReady(z));

		XTime_GetTime(&tEnd);
		//res_hw[i] = XMyproject_Get_layer12_out_0_V(&z);

		//xil_printf("\nDetected HLS peripheral complete. Result received = 0x%X \n",res_hw[i]);
		//xil_printf("\nValid is = 0x%X \n",XMyproject_Get_layer12_out_0_V_vld(&z));
		//xil_printf("\nEnded\n");
		//xil_printf("Clock time:\n");
		//xil_printf("%llu",2*(tEnd - tStart));
		ti = 1.0 * (tEnd - tStart)/200;
		//xil_printf("\nTime in us (fixed point <16,6> hex)\n");
		xil_printf("-%X\n", ti.get_data());
		/*xil_printf("Input Pattern (fixed point <16,6> hex):\n)");

		xil_printf("%X  ",pattern[0]);
		xil_printf("%X  ",pattern[1]);
		xil_printf("%X  ",pattern[2]);
		xil_printf("%X  ",pattern[3]);
		xil_printf("%X  ",pattern[4]);
		xil_printf("%X  ",pattern[5]);
		xil_printf("%X  ",pattern[6]);
		xil_printf("%X  ",pattern[7]);
		xil_printf("%X  ",pattern[8]);
		xil_printf("%X  ",pattern[9]);
		xil_printf("%X  ",pattern[10]);
		xil_printf("%X  ",pattern[11]);
		xil_printf("%X  ",pattern[12]);
		xil_printf("%X  ",pattern[13]);
		xil_printf("%X  ",pattern[14]);
		xil_printf("%X  ",pattern[15]);
		xil_printf("%X  ",pattern[16]);
		xil_printf("%X  ",pattern[17]);
		xil_printf("%X  ",pattern[18]);
		xil_printf("%X  ",pattern[19]);
		xil_printf("%X  ",pattern[20]);
		xil_printf("%X  ",pattern[21]);
		xil_printf("%X  ",pattern[22]);
		xil_printf("%X  ",pattern[23]);
		xil_printf("%X  ",pattern[24]);
		xil_printf("%X  ",pattern[25]);
		xil_printf("%X  ",pattern[26]);
		xil_printf("%X  ",pattern[27]);


		xil_printf("\nOutput(fixed point <17,2> hex):\n");*/
		xil_printf(".%X\n",res_hw[i]);
		xil_printf("\n");
		//xil_printf("Input N. %i\t%X\n",i,res_hw[i]);
		++i;

		delete [] pattern;
	}
	XTime_GetTime(&tEndTot);
	tiTot= (tStartTot-tEndTot)/200;
	xil_printf("\nEnded in %x clocks\n",tiTot.to_uint64());
}


int main()
{
	ap_uint<32>** worddata = database_builder();


	/*float vect[27] = {3 ,4 ,1 ,3 ,0 ,1 ,1 ,1 ,0 ,1 ,12 ,12 ,0 ,0.005859375 ,0.0078125 ,0 ,0 ,1 ,1 ,1 ,0 ,0.002847019826086 ,0.000649754201103 ,0 ,0.002197265625 ,0 ,0};

	ap_fixed<16,6> fip;
	ap_fixed<16,6> fixdata[27];
	XMyproject_Q_dense_20_input_v datatest;
	ap_uint<32> worddata[14];
	for (int i = 0; i < 27; ++i)
	{

		fip = vect[i];
		xil_printf("data %X \t",((ap_uint<16>)fip.range()).to_uint());
		fixdata[i] = fip;
	}
	short k = 0;
	for (int i = 0; i < 26; i+=2)
	{
		xil_printf("k %d fixdata %X \t",k,((ap_uint<16>)fixdata[i].range()).to_uint());
		xil_printf("fixdata+1 %X \t",((ap_uint<16>)fixdata[i+1].range()).to_uint());
		worddata[k] = ((ap_uint<16>)fixdata[i+1].range(),(ap_uint<16>)fixdata[i].range());
		//worddata[j][k] = fixdata[j][i] + fixdata[j][i+1];
		xil_printf("wordata %X \n",worddata[k].to_uint());
		++k;
	}
	xil_printf("\n");
	fip = 0;
	worddata[13] = ((ap_uint<16>)fixdata[26].range(),(ap_uint<16>)fip.range());

	datatest.word_0 = worddata[0].to_uint();
	datatest.word_1 = worddata[1].to_uint();
	datatest.word_2 = worddata[2].to_uint();
	datatest.word_3 = worddata[3].to_uint();
	datatest.word_4 = worddata[4].to_uint();
	datatest.word_5 = worddata[5].to_uint();
	datatest.word_6 = worddata[6].to_uint();
	datatest.word_7 = worddata[7].to_uint();
	datatest.word_8 = worddata[8].to_uint();
	datatest.word_9 = worddata[9].to_uint();
	datatest.word_10 = worddata[10].to_uint();
	datatest.word_11= worddata[11].to_uint();
	datatest.word_12 = worddata[12].to_uint();
	datatest.word_13 = worddata[13].to_uint();

	if(XMyproject_Initialize(&z,NN_DEVICE_ID)!= XST_SUCCESS) return XST_FAILURE;

	xil_printf("\nNN: Accelerator correctly initialized.\n");

	XMyproject_DisableAutoRestart(&z);





	if (XMyproject_IsReady(&z))
		xil_printf("\nHLS peripheral is ready. Starting...\n");
	else {
		xil_printf("\n!!! HLS peripheral is not ready! Exiting...\n");
		return XST_FAILURE;
	}



	XMyproject_Set_q_dense_20_input_V(&z,datatest);
	XMyproject_Start(&z);
	xil_printf("\nHLS peripheral Started\n");
	u32 res_hw;
	do {
		res_hw= XMyproject_Get_layer13_out_0_V(&z);
		} while (!XMyproject_IsReady(&z));

	//res_hw= XMyproject_Get_layer13_out_0_V(&z);

	xil_printf("\nOutput(fixed point <17,2> hex):\n");
	xil_printf(".%X\n",res_hw);
	xil_printf("\n\n");

	xil_printf("\nEnded\n");*/




	if(XMyproject_Initialize(&z,NN_DEVICE_ID)!= XST_SUCCESS) return XST_FAILURE;

		xil_printf("\nNN: Accelerator correctly initialized.\n");

		XMyproject_DisableAutoRestart(&z);





		if (XMyproject_IsReady(&z))
			xil_printf("\nHLS peripheral is ready. Starting...\n");
		else {
			xil_printf("\n!!! HLS peripheral is not ready! Exiting...\n");
			return XST_FAILURE;
		}



















	hw_inference(worddata, &z);

	return 0;
}
