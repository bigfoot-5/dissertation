
(cl:in-package :asdf)

(defsystem "image2position-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Array" :depends-on ("_package_Array"))
    (:file "_package_Array" :depends-on ("_package"))
  ))