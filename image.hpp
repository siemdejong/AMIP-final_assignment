#ifndef IMAGE_HPP_INCLUDED
#define IMAGE_HPP_INCLUDED

#include <map>
#include <vector>
#include <random>
#include <limits>
#include <fstream>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <initializer_list>

/** \brief Cimg.h: header-only C++ library for handling pictures
 *
 * downloaded from https://framagit.org/dtschump/CImg/raw/master/CImg.h
 * it can save images as BMP picturs without requiring extra libraries
 */

#include "CImg/CImg.h"

/** \brief niftiio.hpp uses nifti1_io for NIfTI file I/O
 *         but provides a modern, C++ style interface
 */
#include "nifti/include/niftiio.hpp"

/** \brief NO/DO read data enumeration for easily readable code
 */
enum readdata {
    NO_READ_DATA,
    DO_READ_DATA
};

#ifndef M_E // maybe there's a more general test
#define M_E        2.71828182845904523536   // e 
#define M_LOG2E    1.44269504088896340736   // log2(e) 
#define M_LOG10E   0.434294481903251827651  // log10(e) 
#define M_LN2      0.693147180559945309417  // ln(2) 
#define M_LN10     2.30258509299404568402   // ln(10) 
#define M_PI       3.14159265358979323846   // pi 
#define M_PI_2     1.57079632679489661923   // pi/2 
#define M_PI_4     0.785398163397448309616  // pi/4 
#define M_1_PI     0.318309886183790671538  // 1/pi 
#define M_2_PI     0.636619772367581343076  // 2/pi 
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi) 
#define M_SQRT2    1.41421356237309504880   // sqrt(2) 
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2) 
#endif

#ifndef MIN
#define MIN(a,b) ( (a)<(b)?(a):(b) )
#endif
#ifndef MAX
#define MAX(a,b) ( (a)>(b)?(a):(b) )
#endif
#ifndef SQR
#define SQR(a) ( (a)*(a) )
#endif
#ifndef SIGN
#define SIGN(a) ((a)>0 ? 1 : ((a)<0 ? -1 : 0) )
#endif

// two typedefs used by maxtrees
typedef
	long	level_t;

typedef struct component {
	level_t		level;  										// quantised level
	size_t 		uniq;											// number of points with exactly this level
	size_t		size;    										// number of points with at least this level
	size_t		root;    										// offset of 1st point		
	size_t		parent;											// parent component number
	std::vector <size_t> children;								// components on top of (*this)
	std::vector <size_t> points;								// 1D points (only unique - not children)
	std::map <std::string, std::vector<double_t>> attributes;	// add any number of attributes
} component;



/** \brief aip: namespace for advanced image processing
 *
 *  This namespace contains the imageNd class for
 *  creating and processing multidimansional images.
 */
namespace aip {


/** \brief gauss samples the gauss curve
 *
 * given a position x and a width sigma
 */
template <typename T>
inline T gauss(T sigma, T x) {
    T expVal = - .5* pow( x/sigma, 2);
    T divider = sqrt(2 * M_PI * pow(sigma, 2));
    return (1 / divider) * exp(expVal);
}

/** \brief gausskernel in a vector
 *
 * length of the vector is 3.5 * sigma on either side
 * (gauss is .001 at 3.460871782016046838 times sigma)
 * also possible: 5.1 * sigma
 * (gauss is 10^-6 at 5.07869511287291208 times sigma)
 */
template <typename T>
const std::vector<T> gausskernel(T sigma) {

    double
        limit = 3.5;
    std::vector<T>
        out ( 1 + 2 * (unsigned) ceil( sigma * limit ) );

    // update limit as the 1st value on the x axis
    limit = -ceil (sigma * limit);

    // fill the Gaussian vector, whilst updating the x axis
    for (size_t i=0; i<out.size(); i++)
        out[i] = gauss<T> (sigma, limit++);

    return out;
}

/** \brief filterline for 1-dimensional convolution
 *
 * signal and filter are both of type std::vector
 */
template <typename T, typename U>
void filterline ( std::vector<T>& signal, const std::vector<U>& filter ) {

    // do circular for now -> after signal[size -1] comes signal[0]
    vector <T> workspace ( signal.size() );
    size_t flen2 = filter.size() / 2;

    for ( size_t si = 0; si < signal.size(); si++ )
        for ( size_t fi = 0; fi < filter.size(); fi++ )
            workspace [ (si + flen2) % signal.size() ] += signal [ (si + fi) % signal.size() ] * filter [ fi ];

    std::copy ( workspace.begin(), workspace.end(), signal.begin() );

}



/** \brief class template imageNd for n-dimensional images
 *
 * voxel type T can be any scalar type (char, short, int, float, double)
 */
template <typename T>
class imageNd {

	
	
  public:

    /** \brief use value_type for pixels
     *
     * To be able to use value_type (this name is also
     * used in STL containers) we make it public first
     */
    typedef
    T value_type;



  private:

    /** \brief use value_type for pixels
     *
     * data:    vector with the pixel data
     * sizes:   vector with the dimensions
     * strides: vector with the strides through
     *          the data in each dimension
     * header:  pointer to a NIfTI header
     *          for reading / writing files
     *
     */
    std::vector <size_t>
    sizes;

    std::vector <size_t>
    strides;

    nifti_image
    *header = NULL;

    std::vector <value_type>
    data;
	
	std::vector <component>
		components;

	static constexpr size_t
		undefined = std::numeric_limits<size_t>::max();



  public:

    /** \brief default constructor
     */
    imageNd() {}

    /** \brief destructor
     *
     * destructor: clears vectors and (if
     * required) frees pointer to header
     */
    ~imageNd() {

        data.resize    (0);
        sizes.resize   (0);
        strides.resize (0);
        if ( header != NULL )
            free (header);
    }

    /** \brief constructor for an empty image
     *
     * constructor for an empty image (no
     * nifti header) with given dimensions
     */
    imageNd ( std::initializer_list <size_t> dims ) {

        sizes.resize   ( dims.size()   );
        strides.resize ( dims.size()+1 );

        for ( size_t i=1; i<=sizes.size(); i++ ) {
            sizes [i-1] = dims.begin()[i];
            strides [i] = strides[i-1] * sizes[i-1];
        }

        data.resize ( *strides.rbegin() );

    }

    /** \brief (deep) copy constructor
     *
     * copies data, sizes and header from en existing imageNd
     */
    imageNd ( const imageNd& rhs ) {

        if (rhs.header != NULL)
            header = nifti_copy_nim_info (rhs.header);

        data.resize( rhs.data.size() );
        sizes.resize( rhs.sizes.size() );
        strides.resize( rhs.strides.size() );

        std::copy ( rhs.data.begin(),    rhs.data.end(),    data.begin()    );
        std::copy ( rhs.sizes.begin(),   rhs.sizes.end(),   sizes.begin()   );
        std::copy ( rhs.strides.begin(), rhs.strides.end(), strides.begin() );

    }

    /** \brief assignment operator
     *
     * assigns data, sizes and header of the
     * right-hand side (RHS) image to (*this)
     */
    const imageNd<T>& operator= ( imageNd <T>& rhs ) {

        if ( this != &rhs ) {

            // just to make sure we don't leave stuff
            if ( this->header != NULL ) {
                free ( header );
                header = NULL;
            }

            if (rhs.header != NULL)
                header = nifti_copy_nim_info( rhs.header );

            // just to make sure we don't leave stuff
            data.resize    ( rhs.data.size()    );
            sizes.resize   ( rhs.sizes.size()   );
            strides.resize ( rhs.strides.size() );

            std::copy ( rhs.data.begin(),    rhs.data.end(),    data.begin()    );
            std::copy ( rhs.sizes.begin(),   rhs.sizes.end(),   sizes.begin()   );
            std::copy ( rhs.strides.begin(), rhs.strides.end(), strides.begin() );

        } // if this != rhs

        return *this;

    } // assignment

    /** \brief constructor from a NIfTI image
     *
     * This constructor reads an n-dimensional image
     * ( for NIfTI 1 <= n <= 7 ) from a NIfTI file and
     * assigns its data, sizes and header to (*this).
     */
    imageNd ( std::string filename ) {

        header = nifti_image_read ( filename.c_str(),
                                    DO_READ_DATA );

        sizes.resize   ( header -> dim[0]   );
        strides.resize ( header -> dim[0]+1 );

        // make the array 'strides' so that it uses the last dimension as well
        strides [0] = 1;
        for ( size_t i=1; i<=sizes.size(); i++ ) {
            sizes [i-1] = header -> dim [i];
            strides [i] = strides[i-1] * sizes[i-1];
        }

        data.resize ( *(strides.rbegin()) ); // the end of strides holds the image's size

        aip::getNiftiBricks ( header,
                              header -> data,
                              data.size(),
                              &data );

    }

    /** \brief change the data type for writing a NIfTI file
     *
     * The data type is changed to e.g. float for
     * more flexibility (at the cost of file size).
     */
    void setNIIdatatype ( unsigned dtype ) {
        header-> datatype = dtype;
        nifti_datatype_sizes( header->datatype, &header->nbyper, &header->swapsize ) ;
    }

    /** \brief write to a NIfTI image
     *
     * Writes the contents of (*this) to a NIfTI
     * file using the data and header information.
     * The data type is changed to float for more
     * flexibility (at the cost of file size).
     *
     * NIfTI routines are run on a temporary copy
     *       as they seem to be memory-unsafe
     */
    void saveNII ( std::string filename ) {
        nifti_set_filenames ( header, filename.c_str(), 0, 0 );
        setNiftiBricks      ( header,            &data       );
        nifti_image_write( header );
    }



    /** \brief operator() for positional addressing
     *
     * for an existing image In, this operator can
     * be used with multidimensional co-ordinates to
     * indicate position, so In({x,y,z}) instead of
     * In.data[x + y*sizes[1] + z*sizes[1]*sizes[2]].
     *
     * This operator is for reading only.
     */
    value_type const operator() ( std::initializer_list < size_t > const& indices ) const {
        size_t const offset =
            std::inner_product ( indices.begin(), indices.end(),
                                 strides.begin(),
                                 0 );
        return data [ offset ];
    }

    /** \brief operator() for positional addressing
     *
     * This operator is for modifying the data.
     */
    value_type& operator() ( std::initializer_list < size_t > const& indices ) {
        size_t const offset =
            std::inner_product ( indices.begin(), indices.end(),
                                 strides.begin(),
                                 0 );
        return data [ offset ] ;
    }

    /** \brief operator[] for positional addressing
     *
     * for an existing image In, this operator can
     * be used with multidimensional co-ordinates to
     * indicate position, so In[{x,y,z}] instead of
     * In.data[x + y*sizes[1] + z*sizes[1]*sizes[2]].
     *
     * This operator is for reading only.
     */
    value_type const operator[] ( std::initializer_list < size_t > const& indices ) const {
        size_t const offset =
            std::inner_product ( indices.begin(), indices.end(),
                                 strides.begin(),
                                 0 );
        return data [ offset ];
    }

    /** \brief operator[] for positional addressing
     *
     * This operator is for modifying the data.
     */
    value_type& operator[] ( std::initializer_list < size_t > const& indices ) {
        size_t const offset =
            std::inner_product ( indices.begin(), indices.end(),
                                 strides.begin(),
                                 0 );
        return data [ offset ] ;
    }

    /** brief compute indices at offset
     *
     * inverse of positional addressing - given a position
     * in the 1D data vector, what are its multidimensional
     * indices?
     */
    const std::vector <size_t> indices_at_offset ( size_t pos1d ) {

        auto p = pos1d;
        std::vector <size_t> out ( sizes.size() );

        for ( size_t d = sizes.size()-1; d>0; d-- ) {
            out[d]  =      p / strides[d];
            p      -= out[d] * strides[d];
        }
        out[0] = p;

        return out;

    }


    /** \brief operators += for scalar and imageNd, repectively
     *
     */
    const imageNd<T>& operator+= ( const value_type& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] += rhs;
        return (*this);
    }
    template <typename U>
    const imageNd<T>& operator+= ( const imageNd<U>& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] += rhs.data[s];
        return (*this);
    }

    /** \brief operator + for templated types
     *
     * in this case, types for which += has been defined
     */
    template <typename U>
    const imageNd<T> operator+ ( const U& rhs ) {
        imageNd out(*this);
        out += rhs;
        return out;
    }



    /** \brief operators *= for scalar and imageNd, repectively
     *
     */
    const imageNd<T>& operator*= ( const value_type& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] *= rhs;
        return (*this);
    }
    template <typename U>
    const imageNd<T>& operator*= ( const imageNd<U>& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] *= rhs.data[s];
        return (*this);
    }

    /** \brief operator * for templated types
     *
     * in this case, types for which *= has been defined
     */
    template <typename U>
    const imageNd<T> operator* ( const U& rhs ) {
        imageNd out(*this);
        out *= rhs;
        return out;
    }



    /** \brief operators -= for scalar and imageNd, repectively
     *
     */
    const imageNd<T>& operator-= ( const value_type rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] -= rhs;
        return (*this);
    }
    template <typename U>
    const imageNd<T>& operator-= ( const imageNd<U>& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] -= rhs.data[s];
        return (*this);
    }

    /** \brief operator - for templated types
     *
     * in this case, types for which -= has been defined
     */
    template <typename U>
    const imageNd<T> operator- ( const U& rhs ) {
        imageNd out(*this);
        out -= rhs;
        return out;
    }

    /* \brief operator - without operand
     *
     * negate yourself
     */
    const imageNd<T> operator- ( void ) {
        imageNd out(*this);
        out *= -1;
        return out;
    }


    /** \brief operators /= for scalar and imageNd, repectively
     *
     */
    const imageNd<T>& operator/= ( const value_type rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] /= rhs;
        return (*this);
    }
    template <typename U>
    const imageNd<T>& operator/= ( const imageNd<U>& rhs ) {
        for (size_t s = 0; s < data.size(); s++ )
            data[s] /= rhs.data[s];
        return (*this);
    }

    /** \brief operator / for templated types
     *
     * in this case, types for which /= has been defined
     */
    template <typename U>
    const imageNd<T> operator/ ( const U& rhs ) {
        imageNd<T> out(*this);
        out /= rhs;
        return out;
    }



    /** \brief reciprocal function, returns 1/c
     *
     * Return an image with 1/c for all coefficients c.
     * This function is used for dividing by an image.
     */
    const imageNd<T>& reciprocal ( const imageNd<T>& rhs ) {
        imageNd<T> out(*rhs);
        for (size_t s = 0; s < out.data.size(); s++ )
            out.data[s] = 1 / out.data[s];
        return out;
    }



    /** \brief getsize() - returns the dimensions
     *
     * return dimensions as a std::vector <size_t>
     */
    std::vector<size_t> getsize (          ) {
        return sizes;
    }



    /** \brief getsize() - returns one dimension
     *
     * return dimension given its index in sizes
     */
    size_t              getsize ( size_t s ) {
        return sizes[s];
    }



    /** \brief getdatasize() - returns the number of intensities
     *
     * the size of the vector 'data'
     */
    size_t              getdatasize ( ) {
        return std::accumulate( sizes.begin(), sizes.end(), 1, std::multiplies<size_t>() );
    }



    /** \brief getdata_ptr() - returns the address of the data vector
     *
     * this is a pointer to a vector -- use with care
     */
    std::vector <value_type>*          getdata_ptr ( ) {
        return &data;
    }



    /** \brief getdata_array() - returns the address of the data vector
     *
     * this is a pointer to a vector -- use with care
     */
    value_type*                        getdata_array ( ) {
        return &data[0];
    }



    /** \brief reshape() - change dimensions
     *
     * Changes the sizes and strides vectors.
     * Only works if total #elements does not change.
     */
    void reshape ( std::initializer_list < size_t > const& newsizes ) {

        if ( std::accumulate( newsizes.begin(), newsizes.end(), 1, std::multiplies<size_t>() ) ==
                std::accumulate(    sizes.begin(),    sizes.end(), 1, std::multiplies<size_t>() )
           ) {

            sizes=newsizes;
            for ( size_t i=1; i<=sizes.size(); i++ ) {
                strides [i] =  strides[i-1] * sizes[i-1];
            }

        } else

            std::cout << "reshape impossible because it would change image size";

    } // reshape



    /** \brief addNormalNoise() add normally distributed noise
     *
     * Uses standard library functions to produce random numbers
     * Parameters mu and sigma are doubles, used in the expected way
     */
    void addNormalNoise ( double mu, double sigma ) {

        // random device class instance, source of 'true' randomness for initializing random seed
        std::random_device randomdevice{};

        // Mersenne twister PRNG, initialized with seed from previous random device instance
        std::mt19937 engine { randomdevice() };

        // set up the distribution
        std::normal_distribution <double> normaldist ( mu, sigma );

        // add noise to the data
        // N (mu, sigma) all with <double> values
        // (non-float leads to unexpected behaviour)
        for ( size_t i = 0; i < data.size(); i++ )
            data[i] += normaldist ( engine );

    }

    /** \brief addRicianNoise() add Rician distributed noise
     *
     * Uses standard library functions to produce random numbers
     * Parameters mu and sigma are doubles, used in the expected way
     * for two normal distributions.
     */
    void addRicianNoise ( double mu, double sigma ) {

        // random device class instance, source of 'true' randomness for initializing random seed
        std::random_device randomdevice{};

        // Mersenne twister PRNG, initialized with seed from previous random device instance
        std::mt19937 engine { randomdevice() };

        // set up the distribution
        std::normal_distribution <double> normaldist ( mu, sigma );

        // add Rician noise to the data using 2 normally distributed noise values
        for ( size_t i = 0; i < data.size(); i++ ) {

            double n1 = data[i] + normaldist ( engine );
            double n2           = normaldist ( engine );
            data [i] = sqrt ( ( n1 * n1 ) + ( n2 * n2 ) );

        }

    }



    /** \brief get3Dline() returns a 3D line (along a dimension) from a volume
     *
     * Result is a std::vector of value_type
     *
     * The line is sampled along dimension dim ( 0, 1 or 2 respectively)
     * at position pos1, pos2 in the other dimensions
     * ( 1 and 2, 0 and 2, or 0 and 1, respectively).
     */
    const std::vector <value_type> get3Dline ( size_t dim, size_t pos1, size_t pos2,
            size_t linestart = 0, size_t lineend = UINT_MAX ) {

        std::vector <value_type> out;

        if (sizes.size() != 3) {

            std::cout << "3D lines must be selected from a 3D image\n";

        } else {

            size_t
            step    = strides[dim],
            slicex  = (dim>0) ? 0 : 1,
            slicey  = (dim>1) ? 1 : 2,
            line0   = std::max <size_t> ( linestart, 0          ),
            line1   = std::min <size_t> ( lineend,   sizes[dim] );

            value_type*
            dptr    = data.data() + pos1 * strides[slicex] + pos2 * strides[slicey];

            out.resize ( line1-line0 );

            for ( size_t i = line0; i < line1; i++, dptr+=step )
                out[i] = *dptr;

        } // if sizes

        return out;

    } // get3Dline



    /** \brief get3Dline() puts a 3D line (along a dimension) in a volume
     *
     * Input is a std::vector of value_type
     *
     * The line is inserted along dimension dim ( 0, 1 or 2 respectively)
     * at position pos1, pos2 in the other dimensions
     * ( 1 and 2, 0 and 2, or 0 and 1, respectively).
     */
    void set3Dline ( std::vector <value_type>& in,
                     size_t dim, size_t pos1, size_t pos2,
                     size_t linestart = 0, size_t lineend = UINT_MAX ) {

        if (sizes.size() != 3) {

            std::cout << "3D lines must be selected from a 3D image\n";

        } else {

            size_t
            step = strides[dim],
            slicex = (dim>0) ? 0 : 1,
            slicey = (dim>1) ? 1 : 2,
            line0 = std::max <size_t> ( linestart, 0          ),
            line1 = std::min <size_t> ( line0 + lineend, sizes[dim] );

            value_type* dptr = data.data() + pos1 * strides[slicex] + pos2 * strides[slicey];

            for ( size_t i = line0; i < line1; i++, dptr+=step )
                *dptr = in[i];

        } // if sizes

    } // set3Dline



    /** \brief getSlice() returns a 2D slice from a volume
     *
     * Result is an imageNd of value_type
     * which is a copy of slice no. <sli>
     * along dimension <dim> (0, 1, or 2)
     * and at position <sli>.
     */
    const imageNd<value_type> getSlice( size_t dim, size_t sli, std::string filename="" ) {
        // get a slice from a volume
        // optional: write it out as a .bmp file
        //

        imageNd<value_type> out;

        if (sizes.size() != 3) {

            std::cout << "slices can only be selected from a 3D image\n";

        } else {

            // slice sizes are called slicex (lowest 3D dim index) and slicey (highest)
            size_t slicex = (dim>0) ? 0 : 1;
            size_t slicey = (dim>1) ? 1 : 2;

            // set sizes for sizes, strides and data start with 3D
            out.sizes     = {    sizes[slicex], sizes[slicey], 1                                             };
            out.strides   = { 1, sizes[slicex], sizes[slicex] * sizes[slicey], sizes[slicex] * sizes[slicey] };
            out.data.resize (    *out.strides.rbegin()                                                       );

            // fill the slice by calling get3Dline (from folume) for each y line in the slice
            // loop over highest (outer == slower with largest strides) dimension first
            //
            // dim x -> yz slice, slicex 1, slicey 2 -> lines in y, loop over z -> line pos [ sli z ] = [  sli ypos ]
            // dim y -> xz slice, slicex 0, slicey 2 -> lines in x, loop over z -> line pos [ sli z ] = [  sli ypos ]
            // dim z -> xy slice, slicex 0, slicey 1 -> lines in x, loop over y -> line pos [ y sli ] = [ ypos  sli ]
            for ( size_t ypos=0; ypos<sizes[slicey]; ypos++ ) {

                // position where the line is taken:
                //
                // an x line is taken from an y,z position
                size_t linx = ( slicey>1 ) ?  sli : ypos;
                size_t liny = ( slicey>1 ) ? ypos :  sli;

                std::vector<value_type> sli_line = get3Dline( slicex, linx, liny );
                out.set3Dline ( sli_line, 0, ypos, 0); // x line (0), put in y position liny, 'z' position 0

            } // for ypos

        } // if sizes

        if ( !filename.empty() ) {

            cimg_library::CImg<value_type>*
            my_bitmap = new cimg_library::CImg<value_type> (out.data.data(),
                    out.sizes[0],
                    out.sizes[1],
                    1, 1, true);
            my_bitmap->rotate(180);
            my_bitmap->save_bmp( filename.c_str() );
            delete ( my_bitmap );

        } // if filename

        out.reshape( { out.sizes[0], out.sizes[1] } ); // remove dimension 3 (which is 1) of the output slice
        return out;

    } // getSlice

    /** \brief setSlice() insert a 2D slice into a volume
     *
     * Slice no. <sli> along
     * dimension <dim> (0, 1, or 2)
     * of the current object is copied from the input
     */
    void setSlice ( imageNd<value_type>& input, size_t dim, size_t sli ) {

        if (sizes.size() != 3) {

            std::cout << "slices can only be inserted into a 3D image\n";

        } else {

            // slice sizes are called slicex (lowest 3D dim index) and slicey (highest)
            size_t slicex = (dim>0) ? 0 : 1;
            size_t slicey = (dim>1) ? 1 : 2;

            // check if input sizes match the current imagNd dimensions
            if ( ( input.sizes[0] != sizes[slicex] ) | ( input.sizes[1] != sizes[slicey] ) ) {

                std::cout << "input slice deimensions do not match volume slice size \n";

            }

            // briefly make our slice 3D for using get3Dline()
            input.reshape ( input.sizes[0], input.size[1], 1 );

            for ( size_t ypos=0; ypos<sizes[slicey]; ypos++ ) {

                size_t linx = ( slicey>1 ) ?  sli : ypos;
                size_t liny = ( slicey>1 ) ? ypos :  sli;

                std::vector<value_type> sli_line = input.get3Dline ( sli_line, 0, ypos, 0);
                set3Dline( slicex, linx, liny );

            } // for ypos

            // make our slice 2D again
            input.reshape ( input.sizes[0], input.size[1] );

        } // if sizes

    } // getSlice



    /** \brief getsubvolume() returns a 3D subvolume from a 3D image
     *
     * Ranges are given as xmin, xmax, ymin, ymax, zmin, zmax
     * Input and output are both of type imageNd, and this
     * routine works exclusively with 3D data
     */
    const imageNd <value_type> getsubvolume (size_t startx, size_t endx,
            size_t starty, size_t endy,
            size_t startz, size_t endz) {

        imageNd <value_type> out;

        if (sizes.size() != 3) {

            std::cout << "subvolumes can only be selected from a 3D image\n";

        } else {

            size_t  x0 = std::max<size_t> ( startx, 0 ),
                    y0 = std::max<size_t> ( starty, 0 ),
                    z0 = std::max<size_t> ( startz, 0 ),
                    x1 = std::min<size_t> ( endx, sizes[0] ),
                    y1 = std::min<size_t> ( endy, sizes[1] ),
                    z1 = std::min<size_t> ( endz, sizes[2] );

            out.sizes   = { std::max<size_t> (x1 - x0, 1),
                            std::max<size_t> (y1 - y0, 1),
                            std::max<size_t> (z1 - z0, 1)
                          };
            out.strides = { 1, out.sizes[0], out.sizes[1] * out.sizes[0], out.sizes[2] * out.sizes [1] * out.sizes[0] };
            out.data.resize( *out.strides.rbegin() );

            value_type *dptr = out.data.data();

            for ( size_t z=z0; z<z1; z++ )
                for ( size_t y=y0; y<y1; y++ )
                    for ( size_t x=x0; x<x1; x++ )
                        *dptr++ = operator[] ( { x, y, z } );

        } // if sizes

        return out;

    } // getsubvolume

    /** \brief setsubvolume() inserts a 3D subvolume into a 3D image
     *
     * Ranges are given as xmin, ymin, zmin for where to insert
     * Source and destination are both of type imageNd, and this
     * routine works exclusively with 3D data
     */
    void setsubvolume ( imageNd <value_type>& in,
                        size_t startx,
                        size_t starty,
                        size_t startz ) {

        if ( (sizes.size() != 3) | (in.sizes.size() != 3) ) {

            std::cout << "only 3D can be put in only 3D images\n";

        } else {

            size_t  x0 = std::max<size_t> ( startx, 0 ),
                    y0 = std::max<size_t> ( starty, 0 ),
                    z0 = std::max<size_t> ( startz, 0 ),
                    x1 = std::min<size_t> ( startx + in.sizes[0], sizes[0] ),
                    y1 = std::min<size_t> ( starty + in.sizes[1], sizes[1] ),
                    z1 = std::min<size_t> ( startz + in.sizes[2], sizes[2] );

            value_type *dptr = &in.data[0];

            for ( size_t z=z0; z<z1; z++ )
                for ( size_t y=y0; y<y1; y++ )
                    for ( size_t x=x0; x<x1; x++ )
                        operator[] ({ x, y, z }) = *dptr++;

        } // if sizes

    } // getsubvolume
	
	
	
	/** \brief filter() filter along dimension {0, 1 or 2} in a 3D image
     *
     * The filter is given as a numrical std::vector
     * this method uses the function filterline
     * (outside this class)
     */
    void filter ( std::vector<double> filt, size_t dim ) {

        if (sizes.size() != 3) {

            std::cout << "currently filter works only for 3D images\n";

        } else {

            size_t slicex = (dim>0) ? 0 : 1;
            size_t slicey = (dim>1) ? 1 : 2;

            for ( size_t posx = 0; posx < sizes[slicex]; posx++ )

                for ( size_t posy = 0; posy < sizes[slicey]; posy++ ) {

                    std::vector <value_type> sign = get3Dline( dim, posx, posy );
                    filterline ( sign, filt );
                    set3Dline( sign, dim, posx, posy);

                } // for posy


        } // if sizes

    } //filter



    /** brief compute if two neighbours are valid
     *
     * inverse of positional addressing - given a position
     * in the 1D data vector, what are its multidimensional
     * indices?
     */
    inline bool valid_neighbours ( long offset1, long offset2, long delta = 1 ) {

		long
			off1 = offset1, 
			off2 = offset2,
			pos1, pos2,
			pd = 0;
		bool valid = true;
			
        for ( long d = sizes.size()-1; d>=0; d-- ) {
			
            pos1  = off1 / strides[d];
            pos2  = off2 / strides[d];		
			off1 -= pos1 * strides[d];
			
			if (  ( ( pos1 - pos2 ) >  delta ) ||
				  ( ( pos1 - pos2 ) < -delta ) ) {
					  
				valid = false;
				break;
				
			} else {

				off2  -= pos2 * strides[d];
				
				if ( pos1 != pos2 ) 
					pd++;
				
				if ( pd > delta ) {
					valid = false;
					break;
				}	
			
			} // if pos
					
		} // for d

        return ( valid );

    }
	
    /** \brief GetNNeigbours - return the number of neighbours
     *
     * Ranges are given as xmin, ymin, zmin for where to insert
     * Source and destination are both of type imageNd, and this
     * routine works exclusively with 3D data
     */
    int GetNNeigbours(int ip, int* NeighbourIndices, int ndim, size_t* dimensions) {
        if(ndim<=0 || ndim>3) {
            std::cout<<"ERROR: MImage::GetNNeigbours(). ndim (="<< ndim <<") out of range. \n";
            return 0;
        }
        if(NeighbourIndices==NULL) {
            std::cout<<"ERROR: MImage::GetNNeigbours(). Invalid NULL argument. \n";
            return 0;
        }

        int dimx  = dimensions[0];
        int dimy  = dimensions[1];
        int dimz  = dimensions[2];
        int dimxy = dimx*dimy;

        // Test for out of range
        int ix = ndim>0 ? (  ip    % dimx       ) : 0;
        int iy = ndim>1 ? (((ip-ix)/ dimx)%dimy ) : 0;
        int iz = ndim>2 ? (  ip    / dimxy      ) : 0;

        if((ndim>0 && (ix<0 || ix>=dimx))  ||
                (ndim>1 && (iy<0 || iy>=dimy))  ||
                (ndim>2 && (iz<0 || iz>=dimz))) {
            std::cout<<"ERROR: MImage::GetNNeigbours(). point index out of range (ix, iy, iz) = ("<< ix <<", "<< iy <<" "<< iz << ")\n";
            return 0;
        }

        int NNeig = 0;
        if(ndim>0 && dimx>1) {
            if(ix>0     )
                NeighbourIndices[NNeig++]=ip-1;
            if(ix<dimx-1)
                NeighbourIndices[NNeig++]=ip+1;
        }
        if(ndim>1 && dimy>1) {
            if(iy>0     )
                NeighbourIndices[NNeig++]=ip-dimx;
            if(iy<dimy-1)
                NeighbourIndices[NNeig++]=ip+dimx;
        }
        if(ndim>2 && dimz>1) {
            if(iz>0     )
                NeighbourIndices[NNeig++]=ip-dimxy;
            if(iz<dimz-1)
                NeighbourIndices[NNeig++]=ip+dimxy;
        }
        return NNeig;
    }



    /** \brief GetWatershedImage
     *
	 * Turns an image into its watershed representation
     */
    bool GetWatershedImage() {

        if ( sizes.size() > 3 ) {
            std::cout<<"ERROR: MImage::GetWatershedImage(). Invalid dimensionality ("<< sizes.size() <<"). \n";
            return false;
        }

        value_type max_value=*std::max_element(data.begin(),data.end());
        value_type min_value=*std::min_element(data.begin(),data.end());
        size_t ndim=sizes.size();
        size_t NP=getdatasize();

        std::vector<int>
        Index   (NP),
                Dist    (NP),
                Label   (NP),
                Hist    (max_value + 2),
                CHist   (max_value + 2),
                NeigArr (200);

        // check if image needs inverting and do so if yes
        // (if pixel [0] has lower value than maximum/2 ?)
        if ( data[0] < (max_value/2) ) {
            std::cout << "inverting ... \n";
            for ( size_t i=0; i< NP; i++ )
                data [i] = max_value - min_value - data[i];
        }

        // build the histogram
        for (unsigned n=0; n < NP; n++)
            Hist[data[n]]++;

        // build the cumulative histogram (differs from histogram after index 0)
        for (int k=1; k < max_value+1; k++)
            CHist[k] = CHist[k-1] + Hist[k-1];

        // label point based on value in cumulative histogram -- increasing index to number within intensity
        for (unsigned n=0; n < NP; n++)
            Index[CHist[data[n]]++] = n;

        // subtract histogram from cumulative after labelling
        for (int k=0; k< max_value+1; k++)
            CHist[k] -= Hist[k]; // restore cumulative histogram

        CHist[max_value+1] = NP; // this was still 0

        const int LABELINIT  =   -1;
        const int MASK       =   -2;
        const int WSHED      =    0;
        const int FICTITIOUS =   -3;

        // initialise labels
        for ( unsigned n=0; n< NP; n++)
            Label[n] = LABELINIT;

        std::queue<int> fifoQueue;
        int curlab = 0;

        // Geodesic SKIZ of level h-1 inside level h. INCLUDE LAST LEVEL!
        for( value_type h = min_value; h<=max_value; h++) {
            for( int pixelIndex = CHist[h]; pixelIndex < CHist[h+1]; pixelIndex++) { //mask all pixels at level h
                int   ip  = Index[pixelIndex];
                Label[ip] = MASK;

                int NNEig = GetNNeigbours(ip, NeigArr.data(), ndim, sizes.data());

                for(int i=0; i<NNEig; i++) {
                    if(Label[NeigArr[i]] < 0 && Label[NeigArr[i]] != WSHED)
                        continue;

                    Dist[ip] = 1;  //Initialise queue with neighbours at level h of current basins or watersheds
                    fifoQueue.push(ip);
                    break;
                }
            }

            int curdist = 1;
            fifoQueue.push(FICTITIOUS);

            while(true) { // extend basins
                int voxelIndex = fifoQueue.front();
                fifoQueue.pop();

                if(voxelIndex == FICTITIOUS) {
                    if(fifoQueue.empty())
                        break;

                    fifoQueue.push(FICTITIOUS);
                    curdist++;
                    voxelIndex = fifoQueue.front();
                    fifoQueue.pop();
                }

                int NNEig = GetNNeigbours(voxelIndex, NeigArr.data(), ndim, sizes.data());
                for(int i=0; i<NNEig; i++) { // Labelling p by inspecting neighbours
                    if(Dist[NeigArr[i]] < curdist && (Label[NeigArr[i]] > 0 || Label[NeigArr[i]]==WSHED)) {
                        if(Label[NeigArr[i]] > 0) { // q belongs to an existing basin or to a watershed
                            if(Label[voxelIndex] == MASK || Label[voxelIndex] ==WSHED)
                                Label[voxelIndex] = Label[NeigArr[i]]; // Removed from original algorithm || p.isLabelWSHED() )
                            else if(Label[voxelIndex] != Label[NeigArr[i]])
                                Label[voxelIndex] = WSHED;

                        } // end if lab>0
                        else if (Label[voxelIndex]==MASK)
                            Label[voxelIndex] = WSHED;
                    } else if(Label[NeigArr[i]]==MASK && Dist[NeigArr[i]]==0) {
                        Dist[NeigArr[i]] = curdist + 1;   //q is plateau pixel
                        fifoQueue.push(NeigArr[i]);
                    }
                } // end for, end processing neighbours
            } // end while (loop)

            // Detect and process new minima at level h
            for(int pixelIndex = CHist[h]; pixelIndex < CHist[h+1]; pixelIndex++) { //mask all pixels at level h
                int ip   = Index[pixelIndex];
                Dist[ip] = 0;       // Reset distance to zero

                if(Label[ip]!=MASK)
                    continue;
                curlab++;       // The pixel is inside a new minimum , create new label
                fifoQueue.push(ip);
                Label[ip] = curlab;

                while(fifoQueue.size()) {
                    int voxelIndex = fifoQueue.front();
                    fifoQueue.pop();

                    int NNEig = GetNNeigbours(voxelIndex, NeigArr.data(), ndim, sizes.data());  // replaced ip by voxelIndex
                    for(int i=0; i<NNEig; i++) { // inspect neighbours of q
                        if(Label[NeigArr[i]]!=MASK)
                            continue;

                        fifoQueue.push(NeigArr[i]);
                        Label[NeigArr[i]] = curlab;
                    }
                } // end while
            } // end for
        } // loop over h

        int MINS = (1<<15) -1;
        for ( unsigned i=0; i<NP; i++)
            data[i] = short(MIN(MINS, Label[i]));

        return true;
    }



	/** \brief getNeighbours function, returns vector of neighbour offsets
	 * 
	 * input: connectivity: integer
	 * 			2D images/3D slices:	4	-> neighbours left, right, before, after
     *									8 	-> neighbours of 4, and 4  diagonals
	 * 			3D images				6	-> neighbours left, right, before, after, below, above
	 * 									26	-> neighbours of 6, and 20 diagonals
     */
	std::vector<long long> getNeighbours ( unsigned connectivity = 4 ) {
		
		std::vector<long long> 
			nvec;
			
		switch ( connectivity ) {
			
			case 4:
				assert ( sizes.size() >= 2 );
				nvec = { -strides[1], -1, 1, strides[1] };
				break;
			case 8:
				assert ( sizes.size() >= 2 );
				nvec = { -strides[1] - 1, -strides[1], -strides[1] + 1, 
									 - 1,							 1,
						  strides[1] - 1,  strides[1],  strides[1] + 1
					   };
				break;
			case 6:
				assert ( sizes.size() >= 3 );
				nvec = { -strides[2], -strides[1], -1, 1, strides[1], strides[2] };
				break;
			case 26:
				assert ( sizes.size() >= 3 );
				nvec = { -strides[2] - strides[1] - 1, -strides[2] - strides[1], -strides[2]  -strides[1] + 1,
						 -strides[2] - 1,              -strides[2],							  -strides[2] + 1, 
						 -strides[2] + strides[1] - 1, -strides[2] + strides[1], -strides[2] + strides[1] + 1, 
									 - strides[1] - 1,				-strides[1],			  -strides[1] + 1, 
												   -1,														1, 
									   strides[1] - 1,				 strides[1], 			   strides[1] + 1, 
						  strides[2] - strides[1] - 1,  strides[2] - strides[1],  strides[2] - strides[1] + 1, 
						  strides[2] - 1, 				strides[2],				  strides[2]			  + 1, 
						  strides[2] + strides[1] - 1,	strides[2] + strides[1],  strides[2] + strides[1] + 1
					   };
				break;
				
		} // switch connectivity

		return ( nvec );

	}
	
	
	
	/** \brief getMaxtree function, computes the maxtree of the image
	 * 
	 * Using the right neighbours for the connectivity value provided,
	 * 		 this function fills the components vector, which can then
	 * 		 be used to extract thei pixel representations.
	 * 
     */
	bool getMaxtree (	const level_t levels = UINT16_MAX,
						const char connectivity = 6,
						const std::string method = "Berger" ) {

		////////////////////////////////////////////////////////////////////////////////
		//
		// offsets in the image for connectivities
		// assuming strides in 2d / 3d images
		//	2D		 4: only horizontal and vertical neighbours
		//			 8: horizontal, vertical and diagonal neighbours
		//	3D		 6: only horizontal, vertical or sideways neighbours
		//			26: all horizontal, vertical sideways and combinations
		//
		auto 
			neighbours = getNeighbours ( connectivity );

		auto 
			mn = *std::min_element ( data.begin(), data.end() ), 
			mx = *std::max_element ( data.begin(), data.end() );

		// show neighbours, min and max
		// std::cout << "neighbour offsets:\n"; for ( auto n: neighbours ) std::cout << n << " "; std::cout << "\n"; 
		// std::cout << "min and max: " << mn << ", " << mx << "\n";

		////////////////////////////////////////////////////////////////////////////////
		//
		// quantise the image, store in "quant"
		//

		std::vector <level_t> 
			quant ( data.size(), 0 );
		std::vector <size_t> 
			cdata ( data.size(), 0 );

		if ( ( std::is_same<value_type, float>::value ) || ( std::is_same<value_type, float>::value ) )
			// proper quantisation for floating point data types -- otherwise too many checks needed
			std::transform ( data.begin(), data.end(), quant.begin(),
								[&mn, &mx, &levels] ( auto input ) { return ( input - mn ) * levels / ( mx - mn ); } );
		else
			// simpler algorithm for integers: first check case if our image fits in "levels" - in which case just copy
			if ( ( mx - mn ) < levels )
			std::transform ( data.begin(), data.end(), quant.begin(), 
								[&mn] ( auto input ) { return ( input - mn );	} );
		else {
			// if the image contains more unique intensity levels than "levels", divide by the proper scalar
			auto factor = static_cast<float> ( mx - mn ) / levels;
			std::transform ( data.begin(), data.end(), quant.begin(), 
								[&mn, &factor] ( auto input ) { return ( ( input - mn ) / factor ); } );
		}

		////////////////////////////////////////////////////////////////////////////////
		//
		// sort sorted intensities in the data vector "sorted"
		// and also keep an array "indices" to see where they were in the image
		//

		std::vector<size_t> indices ( data.size() );

		std::iota ( indices.begin(), indices.end(), 0 );      // fill with 0, 1, 2, ..
		std::stable_sort ( indices.begin(), indices.end(),    // 'data' sorted for determining 'indices'
						   [&] ( size_t i, size_t j ) { return ( quant[i] < quant[j] ); } );

		// the parent vector
		std::vector<size_t> parent ( quant.size(), undefined );

		////////////////////////////////////////////////////////////////////////////////
		//
		// build the max-tree from the quantised image
		// using the method from Berger's ICIP 2007 paper
		// http://dx.doi.org/10.1109/ICIP.2007.4379949
		//

		if ( method == "Berger" ) {

			std::vector<size_t>	zpar  ( quant.size(), undefined );
			std::vector<size_t>	root  ( quant.size(),		  0 );
			std::vector<size_t>	rank  ( quant.size(),		  0 );
			std::vector<bool> visited ( quant.size(),	  false );

			// std::string letters = "CDHAFBIGEJ"; // from Berger's 2007 ICIP paper
			for ( size_t i = 0; i < indices.size(); i++ ) { 

				size_t 
					p     = indices[ indices.size() - i -1 ]; // point at index, step from top - bottom
								
				parent	[ p ] = p;          // pixel at this (higher level) starts as parent
				zpar	[ p ] = p;          //							as union-find parent
				root	[ p ] = p;
				visited	[ p ] = true;
								
				auto x    = p;          // keep this as zpar

				// this is valid for Bergers 2007 ICIP example
				// std::cout << "level " << quant[p] << ", procesing node " << p << " (" << letters[p] << ")" <<
				// std::endl;			

				for ( unsigned k = 0; k < neighbours.size(); k++ ) {	// k: neighbour offset
					long long n = p + neighbours[k];					// n: storage position of neighbour

					if ( ( n > -1 )											&& 
					     ( ( static_cast<size_t> ( n ) ) < indices.size() ) && 
						 this->valid_neighbours ( n, p, 1 ) ) {

						if ( visited [ n ] ) {					// if n has been visited it has a zpar

							// this is valid for Bergers 2007 ICIP example
							// std::cout << "looking at neighbour " << n << " (" << letters[n] << ")";

							size_t r = static_cast<size_t> ( n );		// r = root: index of neighbour q
							while ( r != zpar[r] ) 						//     whose zpar points to itself
								r = zpar[r];							//     (short version of 'findroot')

							if ( r != x ) {
								
								parent[ root [ r ] ] = p;
								
								if ( rank [ x ] < rank [ r ] ) {
									
									zpar [ x ] = r;
									root [ r ] = p;
									         x = r;
											 
								} else {
									
									zpar [ r ] = p;

									if ( rank [ r ] >= rank [ p ] ) 
										rank [ p ] += 1;										
									
								}
								
								// this is valid for Bergers 2007 ICIP example
								// std::cout << " whose new root is now " << p << " (" << letters[p] << ")";

							} // if r and p need joining

							// std::cout << std::endl;

						} // if n has a root r

					} // if neighbour n in image

				} // for n neighbours n of p

			} // for p

			////////////////////////////////////////////////////////////////////////////////
			//
			// link component 'leaves' to roots
			//
			
			for ( auto p : indices ) {
				auto q = parent[p];
				if ( quant[parent[q]] == quant[q] )
					parent[p] = parent[q];
			} // for pi

			////////////////////////////////////////////////////////////////////////////////
			//
			// identify components ( ≥1 per level ) at each point
			//
			level_t
				ccount  = 0;
			cdata [ indices [ 0 ] ] = 0;
			for ( auto p : indices ) {
				if ( quant [ parent [ p ] ] == quant [ p ] )
					cdata [ p ] = cdata [ parent [ p ] ];
				else
					cdata [ p ] = ++ccount;
			} // for pi
			
			////////////////////////////////////////////////////////////////////////////////
			//
			// store components: { number, intensity, size, pos. root, number parent }
			// --> at this point, 'size' is *only* the number of points with this label
			//
			components.resize ( ccount + 1 );
			for ( size_t p = 0; p < cdata.size(); p++ ) {
				auto  c = cdata [ p ];
				if ( ! 	components [ c ].size ) {
						components [ c ].root   = p;
						components [ c ].level  = quant                    [ p ];
						components [ c ].parent = cdata  [ parent [ root [ p ] ] ];			
				}
				components [ c ].size++;
				components [ c ].points.push_back ( p );
			} 
			
			////////////////////////////////////////////////////////////////////////////////
			//
			// add higher components to lower: size = uniq + all childrens' sizes
			//
			for ( size_t c = 0; c < components.size(); c++ )       // just copy 'size' to 'uniq'
				components [ c ].uniq = components [ c ].size;	
			for ( size_t c = components.size() - 1; c > 0; c-- ) { // add 'uniq's of children to size
				components [ components [ c ].parent ].size += components [ c ].size;
				components [ components [ c ].parent ].children.push_back ( c );
			}
			for ( size_t c = 0; c < components.size(); c++ )       // sort children from ow to high label
				std::reverse ( components [ c ].children.begin(), components [ c ].children.end() );
			
		} // if Berger method used
		
		bool show_components = true;
		
		if ( show_components ) {
			
				ofstream compfile;
				auto filename = std::string ( header->fname );			
				filename = std::regex_replace( filename, std::regex ( ".nii.gz" ), "_components.txt" );
				filename = std::regex_replace( filename, std::regex ( ".nii"    ), "_components.txt" );
				compfile.open ( filename );
			
				for ( size_t c = 0; c < components.size(); c++ ) {
			
					compfile	<< " component "   << std::setfill ( ' ' ) << std::setw ( 3 ) << c 
								<<  " : { size: "  << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].size    
								<<  ", (unique: "  << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].uniq
								<< "), root: "     << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].root   
								<<  ", level: "    << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].level   
								<<  ", parent: "   << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].parent
								// <<  ", children: " << std::setfill ( ' ' ) << std::setw ( 3 ) << components[c].children 
								<< " } \n" 
								<< std::endl;			
								// std:cout	<< "points: " << components [ c ].points << std::endl;			
													
					} // for c
					
				compfile.close();	
				
			} // if show_components
				
		// replace the pixel intensities by their peak component indices
		for ( size_t c = 0; c < components.size(); c++ )
			for ( auto p: components [ c ].points ) data [ p ] = c;
		
		// If all is well
		return true;

	} // getMaxtree

	/** \brief get a component's points
	 *
	 * get the list of points in a component <comp_start> and include up to <comp_end>
	 */
	const std::vector<size_t> getpoints ( size_t comp_start, size_t comp_end = 0, 
										  bool sorted = false, std::vector<level_t> *mylevels = nullptr ) {
		
		std::vector<size_t> 
			mypoints;
		auto 
			chigh = ( ! comp_end ) ? components.size() : comp_end;
		bool
			use_levels = ( mylevels != nullptr ) ? true : false,
			root_level = false;
	
		mypoints.insert ( mypoints.end(), components [ comp_start ].points.begin(), components [ comp_start ].points.end() );

		// if an array of levels is used, initialise and fill it
		if ( use_levels ) {
			if ( !mylevels->size() )
				root_level = true;				
			mylevels->insert ( mylevels->end(), mypoints.size(), components [ comp_start ].level );			
		}
		
		for ( auto c: components [ comp_start ].children )
			if ( c <= chigh ) {
				auto vec = getpoints( c, chigh, sorted, mylevels );
				mypoints.insert ( mypoints.end(), vec.begin(), vec.end() );
			}
		
		// if the points are sorted and levels are used, these need to be sorted too		
		if ( sorted && root_level ) {
			
			if ( ! use_levels )

				std::sort( mypoints.begin(), mypoints.end() );
				
			else {

				std::vector<size_t> indices ( mypoints.size() );
				std::iota ( indices.begin(), indices.end(), 0 );  
				std::stable_sort ( indices.begin(), indices.end(),
					[&] ( size_t i, size_t j ) { return ( mypoints[i] < mypoints[j] ); } );

				for ( size_t i=0; i<mypoints.size(); i++ )
					
					while ( indices [ i ] != i ) {
					
						// indices of the current / target position 
						size_t indi = indices [ i    ];
						size_t  ind = indices [ indi ];
						
						// values at the target position
						size_t  pnt =   mypoints  [ indi ];
						level_t lev = (*mylevels) [ indi ];
	
						// copy current indices and values to target positions
						  indices   [ indi ] =   indi;
						  mypoints  [ indi ] =   mypoints [ i ];
						(*mylevels) [ indi ] = (*mylevels) [ i ];
						
						// copy target indices and values to current positions
						  indices   [ i ] = ind;
						  mypoints  [ i ] = pnt;
						(*mylevels) [ i ] = lev;
						
					} // while 
						
			} // if use_levels

		} // if sorted
		
		return ( mypoints );
		
	} // getpoints

	/** \brief return a bisimage with the intensities of selected components
	 *
	 * sets only the points in the image that belong to certain components
	 * 
	 */
	bool setpoints ( size_t comp_start, size_t comp_end = 0, 
					 bool sort = true, bool use_levels = true ) {
				
		std::vector<level_t> 
			found ( data.size(), 0 );
		std::vector<level_t> 
			mylevels;
		std::vector<level_t> 
			*levptr = ( use_levels ) ? &mylevels : nullptr;
		std::vector<size_t> 
			mypoints = getpoints ( comp_start, comp_end, sort, levptr );
		
		std::cout << "number of points: " << mypoints.size() << "\n";
		
		std::fill ( begin ( data ), end  (data ), 0 );
		size_t counter = 0;
		for ( auto p: mypoints ) // p is a pixel location
			data [ p ] = ( use_levels ) ? mylevels [ counter++ ] : 1;			

		// If all is well
		return ( true );
		
	} // setpoints

	

}; // class

/** \brief overloaded operators +, *, - and / for basic numerical types
  *
  * This makes it possible to not only do imageNd +  <type>
  * but also                               <type> + imageNd, etc.
  */
template <typename T, typename U>
inline imageNd <T> operator+ ( U x, imageNd <T> y) {
    return y             + x;
}
template <typename T, typename U>
inline imageNd <T> operator* ( U x, imageNd <T> y) {
    return y             * x;
}
template <typename T, typename U>
inline imageNd <T> operator- ( U x, imageNd <T> y) {
    return -y            + x;
}
template <typename T, typename U>
inline imageNd <T> operator/ ( U x, imageNd <T> y) {
    return reciprocal(y) * x;
}

// print vectors and other containers to screen
// see https://stackoverflow.com/a/69649396/1793968
template<typename Container, typename = 
    std::enable_if_t<std::is_same_v<std::void_t<
        decltype(static_cast<typename Container::const_iterator (*)(const Container&)>(&std::cbegin)),
        decltype(static_cast<typename Container::const_iterator (*)(const Container&)>(&std::cend))>, void>
        && !std::is_same_v<std::string, Container>>>
std::ostream& operator<<(std::ostream& out, const Container &vec) {
    std::cout << "[ ";
    for(const auto& t: vec){
        std::cout << t << " ";
    }
    std::cout << "] ";
    return out;
}

}; // namespace

#endif // IMAGE_HPP_INCLUDED
