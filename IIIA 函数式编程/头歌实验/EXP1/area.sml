(* 输出一个整数 *)
fun printInt (a:int) = print(Int.toString(a)^" "); 

(* 输出一个实数 *)
fun printReal (a:real) = print(Real.toString(a)^" ");

(* 读取一个整数 *)
fun getInt () = Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
    
(* 读取一个实数 *)    
fun getReal () = Option.valOf (TextIO.scanStream (Real.scan) TextIO.stdIn);
    

(*  完成Begin和End间代码的修改  *)    

(*****Begin*****)

 val t = 5.0; 
 fun area r = t * r;
 val m = getInt ();
print (Int.toString (round(area (Real.fromInt m))));

(*****End*****) 
