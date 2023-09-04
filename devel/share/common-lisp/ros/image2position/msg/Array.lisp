; Auto-generated. Do not edit!


(cl:in-package image2position-msg)


;//! \htmlinclude Array.msg.html

(cl:defclass <Array> (roslisp-msg-protocol:ros-message)
  ((array
    :reader array
    :initarg :array
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass Array (<Array>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Array>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Array)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name image2position-msg:<Array> is deprecated: use image2position-msg:Array instead.")))

(cl:ensure-generic-function 'array-val :lambda-list '(m))
(cl:defmethod array-val ((m <Array>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader image2position-msg:array-val is deprecated.  Use image2position-msg:array instead.")
  (array m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Array>) ostream)
  "Serializes a message object of type '<Array>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'array))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'array))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Array>) istream)
  "Deserializes a message object of type '<Array>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'array) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'array)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Array>)))
  "Returns string type for a message object of type '<Array>"
  "image2position/Array")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Array)))
  "Returns string type for a message object of type 'Array"
  "image2position/Array")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Array>)))
  "Returns md5sum for a message object of type '<Array>"
  "71f1005c81b671681646a574c6360c24")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Array)))
  "Returns md5sum for a message object of type 'Array"
  "71f1005c81b671681646a574c6360c24")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Array>)))
  "Returns full string definition for message of type '<Array>"
  (cl:format cl:nil "float32[] array~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Array)))
  "Returns full string definition for message of type 'Array"
  (cl:format cl:nil "float32[] array~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Array>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'array) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Array>))
  "Converts a ROS message object to a list"
  (cl:list 'Array
    (cl:cons ':array (array msg))
))
