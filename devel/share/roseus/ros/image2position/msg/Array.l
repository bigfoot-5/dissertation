;; Auto-generated. Do not edit!


(when (boundp 'image2position::Array)
  (if (not (find-package "IMAGE2POSITION"))
    (make-package "IMAGE2POSITION"))
  (shadow 'Array (find-package "IMAGE2POSITION")))
(unless (find-package "IMAGE2POSITION::ARRAY")
  (make-package "IMAGE2POSITION::ARRAY"))

(in-package "ROS")
;;//! \htmlinclude Array.msg.html


(defclass image2position::Array
  :super ros::object
  :slots (_array ))

(defmethod image2position::Array
  (:init
   (&key
    ((:array __array) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _array __array)
   self)
  (:array
   (&optional __array)
   (if __array (setq _array __array)) _array)
  (:serialization-length
   ()
   (+
    ;; float32[] _array
    (* 4    (length _array)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _array
     (write-long (length _array) s)
     (dotimes (i (length _array))
       (sys::poke (elt _array i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _array
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _array (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _array i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get image2position::Array :md5sum-) "71f1005c81b671681646a574c6360c24")
(setf (get image2position::Array :datatype-) "image2position/Array")
(setf (get image2position::Array :definition-)
      "float32[] array

")



(provide :image2position/Array "71f1005c81b671681646a574c6360c24")


