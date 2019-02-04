#ifndef __TRI_DIAG_SOLVER_OPS_CL__
#define __TRI_DIAG_SOLVER_OPS_CL__

#ifdef TRIDIAG_COMPLEX


#ifdef TRIDIAG_DOUBLE

double2 clReal(const double val) { return (double2)(val, 0); }

double clAbs(const double2 val) { return length(val); }

double2 clMul(const double2 lhs, const double2 rhs)
{
	return (double2)(lhs.x * rhs.x - lhs.y * rhs.y,
                     lhs.y * rhs.x + lhs.x * rhs.y);
}

double2 clDiv(const double2 lhs, const double2 rhs)
{
	double denom = rhs.x*rhs.x + rhs.y*rhs.y;

    return (double2)((lhs.x*rhs.x + lhs.y*rhs.y) / denom,
                     (lhs.y*rhs.x - lhs.x*rhs.y) / denom);
}

double2 clInv(const double2 val) { return clDiv(clReal(1), val); }

double2 clFma(const double2 x, const double2 y, const double2 d) { return clMul(x, y) + d; }

#else // TRIDIAG_DOUBLE

float2 clReal(const float val) { return (float2)(val, 0); }

float clAbs(const float2 val) { return length(val); }

float2 clMul(const float2 lhs, const float2 rhs)
{
	return (float2)(lhs.x * rhs.x - lhs.y * rhs.y,
                    lhs.y * rhs.x + lhs.x * rhs.y);
}

float2 clDiv(const float2 lhs, const float2 rhs)
{
	float denom = rhs.x*rhs.x + rhs.y*rhs.y;

    return (float2)((lhs.x*rhs.x + lhs.y*rhs.y) / denom,
                    (lhs.y*rhs.x - lhs.x*rhs.y) / denom);
}

float2 clInv(const float2 val) { return clDiv(clReal(1), val); }

float2 clFma(const float2 x, const float2 y, const float2 d) { return clMul(x, y) + d; }

#endif // TRIDIAG_DOUBLE


#else // TRIDIAG_COMPLEX


#ifdef TRIDIAG_DOUBLE

double clReal(const double val) { return val; }

double clAbs(const double val) { return fabs(val); }

double clMul(const double lhs, const double rhs) { return lhs * rhs; }

double clDiv(const double lhs, const double rhs) { return lhs / rhs; }

double clInv(const double val) { return 1.0 / val; }

double clFma(const double x, const double y, const double d) { return fma(x, y, d); }

#else // TRIDIAG_DOUBLE

float clReal(const float val) { return val; }

float clAbs(const float val) { return fabs(val); }

float clMul(const float lhs, const float rhs) { return lhs * rhs; }

float clDiv(const float lhs, const float rhs) { return lhs / rhs; }

float clInv(const float val) { return 1.0f / val; }

float clFma(const float x, const float y, const float d) { return fma(x, y, d); }

#endif // TRIDIAG_DOUBLE


#endif // TRIDIAG_COMPLEX

#endif // __TRI_DIAG_SOLVER_OPS_CL__
