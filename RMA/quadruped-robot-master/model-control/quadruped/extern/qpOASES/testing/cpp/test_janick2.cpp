/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2017 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file testing/cpp/test_janick2.cpp
 *	\author Hans Joachim Ferreau
 *	\version 3.2
 *	\date 2011-2017
 *
 *	Example that causes troubles when hotstarting.
 */


#include <qpOASES.hpp>
#include <qpOASES/UnitTesting.hpp>
#include <stdio.h>

#define __MAKE_POS_DEF__
// #undef __MAKE_POS_DEF__

int main( )
{
	USING_NAMESPACE_QPOASES

	int_t nWSR = 100;
	/* Setting up QProblem object. */
	SQProblem example( 11,3 );

	Options options;
	options.setToFast();
// 	options.setToDefault();
	options.initialStatusBounds = REFER_NAMESPACE_QPOASES ST_INACTIVE;

	//options.terminationTolerance = 1.e-12;
	options.initialStatusBounds = REFER_NAMESPACE_QPOASES ST_INACTIVE;
	//options.enableFarBounds = REFER_NAMESPACE_QPOASES BT_FALSE;
	//options.enableRegularisation = REFER_NAMESPACE_QPOASES BT_FALSE;

	example.setOptions( options );


	/* Setup data of first QP. */
	real_t H[11*11] = {
	 6.20100988531485e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-3.84861756786704e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-7.43268431723266e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 1.00000000000000e-01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-3.84861756786704e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 5.41188294952735e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 4.61304826562310e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.10000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 1.00000000000000e-01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.01000000000000e+01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.10000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-7.43268431723266e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 4.61304826562310e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-1.73544778892019e+01,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+01,
	};
	real_t g[11] =	{
//	 9.13378607947379e-07,
//	 0.00000000000000e+00,
//	 0.00000000000000e+00,
//	-1.12448469735682e-06,
//	 0.00000000000000e+00,
//	 0.00000000000000e+00,
//	 0.00000000000000e+00,
//	 0.00000000000000e+00,
//	-1.18185650936822e+02,
//	 0.00000000000000e+00,
//	 0.00000000000000e+00,
	 -6.93766478421491e-04,
	  3.84943289898669e-04,
	 -3.63779116055460e-05,
	  6.38114176725135e-04,
	  1.85797765355698e-04,
	  6.21922122437904e-05,
	  0.00000000000000e+00,
	  0.00000000000000e+00,
	 -1.18185758699839e+02,
	  1.54357580390960e-05,
	  5.39852809009711e-06,
	};
	real_t zLow[11] =	{
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	-4.50000000000000e+01,
	-1.00000000000000e+12,
	};
	real_t zUpp[11] =	{
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 1.00000000000000e+12,
	 1.00000000000000e+12,
	 0.00000000000000e+00,
	 4.50000000000000e+01,
	 1.00000000000000e+12,
	};
	real_t D[11*3] =	{
	 1.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-1.00000000000000e-02,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-1.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-1.00000000000000e-02,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 1.00000000000000e+00,	 0.00000000000000e+00,	-1.00000000000000e-02,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	};
	real_t dLow[3] =	{
	-1.00000000000000e+12,
	-1.00000000000000e+12,
	-1.00000000000000e+12,
	};
	real_t dUpp[3] =	{
	 2.12376384003361e-01,
	 4.78762361599664e+00,
	 8.95204469622285e-01,
	};


	#ifdef __MAKE_POS_DEF__
// 	H[9*11+9] += 30;
	H[8*11+8] += 30;
	#endif
	returnValue status = example.init( H,g,D,zLow,zUpp,dLow,dUpp, nWSR );
	printf("qpOASES_status = %d\n", (int)status );

	/* Get and print solution of second QP. */
	real_t xOpt[11];
	real_t yOpt[11+3];
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf("first QP:\n");
	for (int_t ii =0; ii<11; ++ii )	{
		printf("x[%d] = %.3e\n", (int)ii, xOpt[ii]);
	}

	/* Compute KKT tolerances */
	real_t stat, feas, cmpl;
	SolutionAnalysis analyzer;

	analyzer.getKktViolation( &example, &stat,&feas,&cmpl );
	printf( "\nstat = %e\nfeas = %e\ncmpl = %e\n", stat,feas,cmpl );

	QPOASES_TEST_FOR_TOL( stat,1e-9 );
	QPOASES_TEST_FOR_TOL( feas,1e-7 );
	QPOASES_TEST_FOR_TOL( cmpl,1e-15 );

	nWSR = 100;

	/* Setup data of second QP. */
	real_t H2[11*11] = {
	 6.20101055067033e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-3.84861780549400e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-7.43268533746787e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 1.00000000000000e-01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-3.84861780549400e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 5.41188396792859e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 4.61304896387257e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.10000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 1.00000000000000e-01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.01000000000000e+01,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.10000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-7.43268533746787e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 4.61304896387257e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-1.73544780086860e+01,	 0.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+00,	 0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 2.00000000000000e+01,
	};
	real_t g2[11] =	{
	-8.92227256391600e-08,
	 6.89531726031141e-08,
	-1.91970120006650e-07,
	 1.77206607789402e-07,
	-3.83145267945144e-09,
	-1.88284265021358e-08,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	-1.18185657392775e+02,
	 1.45337027424899e-17,
	-6.04156175796480e-20,
	};
	real_t zLow2[11] =	{
	-1.07876236566374e+01,
	-1.00000000002784e+12,
	-1.00000000000000e+12,
	-8.30554585107279e-08,
	-7.00000003695781e+00,
	-2.60479531522807e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	 0.00000000000000e+00,
	-4.50000000018062e+01,
	-1.00000000000000e+12,
	};
	real_t zUpp2[11] =	{
	 9.99999999989212e+11,
	 9.99999999972157e+11,
	 1.00000000000000e+12,
	 4.68471853498991e+01,
	 6.99999996304219e+00,
	 9.99999999997395e+11,
	 1.00000000000000e+12,
	 1.00000000000000e+12,
	 0.00000000000000e+00,
	 4.49999999981938e+01,
	 1.00000000000000e+12,
	};
	real_t D2[11*3] =	{
	 1.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	-1.00000000000000e-02,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	-1.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-1.00000000000000e-02,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,	-0.00000000000000e+00,
	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,	 1.00000000000000e+00,	 0.00000000000000e+00,	-1.00000000000000e-02,	 0.00000000000000e+00,	 0.00000000000000e+00,	 0.00000000000000e+00,
	};
	real_t dLow2[3] =	{
	-1.00000000000000e+12,
	-1.00000000000000e+12,
	-1.00000000000000e+12,
	};
	real_t dUpp2[3] =	{
	 2.12376343362616e-01,
	 4.78762365663739e+00,
	 8.95204684771929e-01,
	};
	#ifdef __MAKE_POS_DEF__
	H2[8*11+8] += 30;
// 	H2[9*11+9] += 30;
	#endif


	status = example.hotstart( H2,g2,D2,zLow2,zUpp2,dLow2,dUpp2, nWSR );
	printf("qpOASES_status = %d\n", (int)status );

	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf("second QP:\n");
	for (int_t ii =0; ii<11; ++ii )	{
		printf("x[%d] = %.3e\n", (int)ii, xOpt[ii]);
	}
	
	printf( "\nQP objective value: %.3e\n", example.getObjVal() );

	/* Compute KKT tolerances */
	analyzer.getKktViolation( &example, &stat,&feas,&cmpl );
	printf( "\nstat = %e\nfeas = %e\ncmpl = %e\n", stat,feas,cmpl );

	QPOASES_TEST_FOR_TOL( stat,1e-9 );
	QPOASES_TEST_FOR_TOL( feas,1e-7 );
	QPOASES_TEST_FOR_TOL( cmpl,1e-15 );


	return TEST_PASSED;
}


/*
 *	end of file
 */
