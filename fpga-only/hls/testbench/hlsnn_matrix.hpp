#ifndef HLSNN_MATRIX_HPP
#define HLSNN_MATRIX_HPP

#include <iostream>

namespace hlsnn {

    template<class T, unsigned M, unsigned N>
    class matrix {

    public:
        T m_[M][N];

        matrix() {

            defult_const_row:
            for (unsigned i = 0; i < M; i++) {
            	defult_const_col:
                for (unsigned j = 0; j < N; j++) {
                    m_[i][j] = T(0);
                }
            }
        }


        matrix(bool select) {

        	if(select){
				defult_const_row:
				for (unsigned i = 0; i < M; i++) {
					defult_const_col:
					for (unsigned j = 0; j < N; j++) {
						m_[i][j] = T(0);
					}
				}
        	}
        }

        matrix(T *t){
        	ptr_const_row:
            for (unsigned m = 0; m < M; m++) {
            	ptr_const_col:
                for (unsigned n = 0; n < N; n++) {
                    m_[m][n] = t[n + m * (N)];
                }
            }
        }

        matrix(const matrix &other) {
        	copy_const_row:
            for (unsigned i = 0; i < M; i++) {
            	copy_const_col:
                for (unsigned j = 0; j < N; j++) {
                    m_[i][j] = other(i, j);
                }
            }
        }

        matrix &operator=(matrix other) {
        	copy_assign_row:
            for (unsigned i = 0; i < M; i++) {
            	copy_assign_col:
                for (unsigned j = 0; j < N; j++) {
                    m_[i][j] = other(i, j);
                }
            }

            return *this;
        }

        matrix(T *t, int m, int n) {
            int index_row = 0;
            int index_col = 0;
            for (int i = 0; i < m * n; i++) {
                if ((i % n == 0) && (i > 0)) {
                    index_row++;
                    index_col = 0;
                }
                m_[index_row][index_col] = t[i];
                index_col++;
            }

            if(m < M)
                for(int i = m; i < M; i++){
                    for(int j = 0; j < N; j++){
                        m_[i][j] = 0;
                    }
                }

            if(n < N)
                for(int i = 0; i < M; i++){
                    for(int j = n; j < N; j++){
                        m_[i][j] = 0;
                    }
                }
        }


        void loadData(T *t) {
#pragma HLS INLINE off
        	matrix_loaddata_default:
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                    m_[m][n] = t[n + m * (N)];
                }
            }
        }

        void loadData(T *t, unsigned M_, unsigned N_){
#pragma HLS INLINE off
        	matrix_loaddata:
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	if(m < M_ && n < N_){
                		m_[m][n] = t[n + m * (N_)];
                	}
                }
            }
        }

        void loadData(const T *t, unsigned M_, unsigned N_){
#pragma HLS INLINE off
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	if(m < M_ && n < N_){
                		m_[m][n] = t[n + m * (N_)];
                	}
                }
            }
        }

    	template<class T2, unsigned MaxM, unsigned MaxN>
    	void load_data(T2 (&matrix)[MaxM][MaxN], unsigned M_, unsigned N_) {
#pragma HLS INLINE off
            for (unsigned i = 0; i < M; i++) {
                for (unsigned j = 0; j < N; j++) {
                    m_[i][j] = matrix[i][j];
                }
            }
    	}


        void loadData(const T t[M][N]){
#pragma HLS INLINE off
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	m_[m][n] = t[n + m * (N)];
                }
            }
        }

        template<unsigned MaxM, unsigned MaxN>
        void loadData(matrix<T, MaxM, MaxN> &t, unsigned M_, unsigned N_){
#pragma HLS INLINE off
        	load_data_matrix_template:
            for (unsigned m = 0; m < M_; m++) {
                for (unsigned n = 0; n < N_; n++) {
                    if(m < M_ && n < N_){
                    	m_[m][n] = t.m_[m][n];
                    }
                }
            }
        }

        void loadData(T t[M][N]){
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                    m_[m][n] = t[m][n];
                }
            }
        }


        template<typename T2>
        void copyToArray(T2 tmp[M][N]){
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	tmp[m][n] = T2(m_[m][n]);
                }
            }
        }


        template<typename T2>
        void copyToArray(T2 tmp[M * N]){
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	tmp[n + m * (N)] = T2(m_[m][n]);
                }
            }
        }


        template<unsigned M2, unsigned N2, typename T2>
        void copyToArray(T2 tmp[M2 * N2]){
            for (unsigned m = 0; m < M2; m++) {
                for (unsigned n = 0; n < N2; n++) {
                	tmp[n + m * (N)] = T2(m_[m][n]);
                }
            }
        }

        template<typename T2>
        void copyToArray(T2 tmp[], unsigned M2, unsigned N2){
#pragma HLS INLINE off
            for (unsigned m = 0; m < M; m++) {
                for (unsigned n = 0; n < N; n++) {
                	if(m < M2 && n < N2){
                		tmp[n + m * (N)] = T2(m_[m][n]);
                	}
                }
            }
        }


        T square(){
            T tmp = 0;
            for(int m = 0; m < M; m++){
                for(int n = 0; n < N; n++){
                    tmp += m_[m][n] * m_[m][n];
                }
            }

            return tmp;

        }

        void fill(T v){
#pragma HLS INLINE off
        	matrix_fill:
            for(int m = 0; m < M; m++){
                for(int n = 0; n < N; n++){
                    m_[m][n] = v;
                }
            }
        }

        // Operators
        T &operator()(const unsigned i, const unsigned j) {
            return m_[i][j];
        }

        const T &operator()(const unsigned i, const unsigned j) const {
            return m_[i][j];
        }


        template<class C, unsigned I, unsigned J, unsigned K, unsigned L>
        friend matrix<C, I, L> operator*(const matrix<C, I, J> &m1, const matrix<C, K, L> &m2);

        // Get and Set operators
        T &get(const unsigned i, const unsigned j) {
            return m_[i][j];
        }

        void set(const unsigned i, const unsigned j, T &t) {
            m_[i][j] = t;
        }

        unsigned int getRowSize() {
            return M;
        }

        unsigned int getColSize() {
            return N;
        }

        // Helper operators
        void toArray(T *outA) {
            int index_r = 0;
            int index_c = 0;
            for (int i = 0; i < M * N; ++i) {
                if (i % N == 0 && i > 0) {
                    index_r++;
                    index_c = 0;
                }
                outA[i] = m_[index_r][index_c];
                index_c++;
            }
        }

        // Mathematical operations

        T avg() {
            T avg = (*this).sum();
            avg = avg / (M*N);
            return avg;
        }

        T max() {
            T maxval = m_[0][0];
            row:
            for (int i = 0; i < M; i++) {
                col:
                for (int j = 0; j < N; j++) {
                    if (m_[i][j] > maxval)
                        maxval = m_[i][j];
                }
            }

            return maxval;
        }

        T min() {
            T minval = m_[0][0];
            row:
            for (int i = 0; i < M; i++) {
                col:
                for (int j = 0; j < N; j++) {
                    if (m_[i][j] > minval)
                        minval = m_[i][j];
                }
            }

            return minval;
        }

        T getRowSum(unsigned int r) {
            T sum = T(0);
            col:
            for (int j = 0; j < N; j++) {
                sum += m_[r][j];
            }
            return sum;
        }

        T getColSum(unsigned int c) {
            T sum = T(0);
            row:
            for (int i = 0; i < M; i++) {
                sum += m_[i][c];
            }
            return sum;
        }

        T getRowMax(unsigned int r) {
            T maxval = m_[r][0];
            col:
            for (int j = 0; j < N; j++) {
                if (m_[r][j] > maxval)
                    maxval = m_[r][j];
            }
            return maxval;
        }

        T getColMax(unsigned int c) {
            T maxval = m_[0][c];
            row:
            for (int i = 0; i < M; i++) {
                if (m_[i][c] > maxval)
                    maxval = m_[i][c];
            }
            return maxval;
        }

        T getRowMin(unsigned int r) {
            T minval = m_[r][0];
            col:
            for (int j = 0; j < N; j++) {
                if (m_[r][j] < minval)
                    minval = m_[r][j];
            }
            return minval;
        }

        T getColMin(unsigned int c) {
            T minval = m_[0][c];
            row:
            for (int i = 0; i < M; i++) {
                if (m_[i][c] < minval)
                    minval = m_[i][c];
            }
            return minval;
        }

        bool isEqual(const matrix &other) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (other(i, j) != (*this)(i, j)) {
                        return false;
                    }
                }
            }
            return true;
        }

        hlsnn::matrix<T, N, M> transpose() {
            hlsnn::matrix<T, N, M> result;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    result(j, i) = m_[i][j];
                }
            }
            return result;
        }

        void print() {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    std::cout << m_[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }


        void print(unsigned M_, unsigned N_) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                	if(i < M_ && j < N_){
                		std::cout << m_[i][j] << " ";
                	}
                }
                std::cout << std::endl;
            }
        }


        void printSize() {
            std::cout << "Size: (" << M << "," << N << ")" << std::endl;
        }

        friend hlsnn::matrix<T, M, N> operator*(hlsnn::matrix<T, M, N> &a, T b) {
        	hlsnn::matrix<T, M, N> result;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    result(i, j) = a(i, j) * b;
                }
            }
            return result;
        }

    };

    template<class T, unsigned M, unsigned N>
    std::ostream &operator<<(std::ostream &out, matrix<T, M, N> &m) {
        out << std::endl;
        row:
        for (unsigned i = 0; i < M; i++) {
            col:
            for (unsigned j = 0; j < N; j++) {
                out << m(i, j) << " ";
            }
            out << std::endl;
        }

        return out;
    }

    template<class T, unsigned M, unsigned N>
    matrix<T, M, N> operator+(const matrix<T, M, N> &m1, const matrix<T, M, N> &m2) {
        matrix<T, M, N> out;
        row:
        for (unsigned i = 0; i < M; i++) {
            col:
            for (unsigned j = 0; j < N; j++) {
                out(i, j) = m1(i, j) + m2(i, j);
            }
        }

        return out;
    }

    template<class T, class T2, unsigned M, unsigned N>
    matrix<T, M, N> operator-(const matrix<T, M, N> &m1, const matrix<T2, M, N> &m2) {
        matrix<T, M, N> out;
        for (unsigned i = 0; i < M; i++) {
            for (unsigned j = 0; j < N; j++) {
                out(i, j) = m1(i, j) - T(m2(i, j));
            }
        }

        return out;
    }


    template<class C, unsigned I, unsigned J, unsigned L>
    matrix<C, I, L> operator*(const matrix<C, I, J> &m1, const matrix<C, J, L> &m2) {
        matrix<C, I, L> out;
        row:
        for (unsigned i = 0; i < I; i++) {
            col:
            for (unsigned j = 0; j < L; j++) {
                C sum = 0;
                prod:
                for (unsigned k = 0; k < J; k++) {
                    sum += m1(i, k) * m2(k, j);
                }
                out(i, j) = sum;
            }
        }

        return out;
    }

    template<class C, class T, unsigned M, unsigned N>
    matrix<C, M, N> operator*(const C s1, const matrix<T, M, N> &m1) {
        matrix<T, M, N> out;
        row:
        for (unsigned i = 0; i < M; i++) {
            col:
            for (unsigned j = 0; j < N; j++) {
                out(i, j) = T(s1) * m1(i, j);
            }
        }

        return out;
    }


    template<typename T, unsigned M, unsigned N, unsigned K, unsigned L>
    void matrixMultiply(matrix<T, M, N> &a, matrix<T, K, L> &b, matrix<T, M, L> &result){
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < L; j++) {
            	T tmp = 0;
                for (int k = 0; k < K; k++) {
                	tmp += a(i, k) * b(k, j);
                }

                result(i,j) = tmp;
            }
        }
    }

    template<typename T, unsigned M, unsigned N>
    void scalarMultiply(matrix<T, M, N> &a, T b, matrix<T, M, N> &result) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                result(i, j) = a(i, j) * b;
            }
        }
    }

    template<class T, unsigned M_, unsigned N_>
    void increment(hlsnn::matrix<T, M_, N_> &in){
#pragma HLS inline off
    	T tmp = 0;
    	increment_row:
    	for(int m = 0; m < M_; m++){
    		increment_col:
    		for(int n = 0; n < N_; n++){
    			tmp = in(m, n);
    			in(m, n) = tmp + 1;
    		}
    	}
    }

    template<typename T, unsigned M, unsigned N>
    using tensor2d_type = matrix<T, M, N>;

}


#endif
