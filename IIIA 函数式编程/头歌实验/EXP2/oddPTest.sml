(* 输出一个bool *)
fun printBool (a:bool) = print(Bool.toString(a)^" "); 

(* 读取一个整数 *)
fun getInt () = Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
 
(* oddP : int -> bool *)
(* REQUIRES: n >= 0 *)
(* ENSURES: if n is odd *)
(*  编写函数oddP, 完成Begin和End间代码的修改  *)   

(****   Begin      ****)
fun oddP (0) = false
| oddP (1) = true
| oddP n = oddP (n-2)
(****   End      ****)

val m = getInt ();
printBool (oddP m); 
