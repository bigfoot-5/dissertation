// Generated by gencpp from file image2position/Array.msg
// DO NOT EDIT!


#ifndef IMAGE2POSITION_MESSAGE_ARRAY_H
#define IMAGE2POSITION_MESSAGE_ARRAY_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace image2position
{
template <class ContainerAllocator>
struct Array_
{
  typedef Array_<ContainerAllocator> Type;

  Array_()
    : array()  {
    }
  Array_(const ContainerAllocator& _alloc)
    : array(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _array_type;
  _array_type array;





  typedef boost::shared_ptr< ::image2position::Array_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::image2position::Array_<ContainerAllocator> const> ConstPtr;

}; // struct Array_

typedef ::image2position::Array_<std::allocator<void> > Array;

typedef boost::shared_ptr< ::image2position::Array > ArrayPtr;
typedef boost::shared_ptr< ::image2position::Array const> ArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::image2position::Array_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::image2position::Array_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::image2position::Array_<ContainerAllocator1> & lhs, const ::image2position::Array_<ContainerAllocator2> & rhs)
{
  return lhs.array == rhs.array;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::image2position::Array_<ContainerAllocator1> & lhs, const ::image2position::Array_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace image2position

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::image2position::Array_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::image2position::Array_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::image2position::Array_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::image2position::Array_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::image2position::Array_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::image2position::Array_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::image2position::Array_<ContainerAllocator> >
{
  static const char* value()
  {
    return "71f1005c81b671681646a574c6360c24";
  }

  static const char* value(const ::image2position::Array_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x71f1005c81b67168ULL;
  static const uint64_t static_value2 = 0x1646a574c6360c24ULL;
};

template<class ContainerAllocator>
struct DataType< ::image2position::Array_<ContainerAllocator> >
{
  static const char* value()
  {
    return "image2position/Array";
  }

  static const char* value(const ::image2position::Array_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::image2position::Array_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] array\n"
;
  }

  static const char* value(const ::image2position::Array_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::image2position::Array_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.array);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Array_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::image2position::Array_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::image2position::Array_<ContainerAllocator>& v)
  {
    s << indent << "array[]" << std::endl;
    for (size_t i = 0; i < v.array.size(); ++i)
    {
      s << indent << "  array[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.array[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // IMAGE2POSITION_MESSAGE_ARRAY_H
