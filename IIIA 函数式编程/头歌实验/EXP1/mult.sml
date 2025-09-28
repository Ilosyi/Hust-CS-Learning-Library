fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
    
fun printIntList ( [] ) = ()
  | printIntList ( x::xs ) = 
    let
	val tmp = printInt(x)
    in
	printIntList(xs)
    end;

fun getIntList ( 0 ) = []
  | getIntList ( N:int) = getInt()::getIntList(N-1);
    

(*  完成Begin和End间代码的编写  *)

(*****Begin*****)

(* sum : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: sum(L) evaluates to the sum of the integers in L. *)
fun sum [ ] = 0
    | sum (x ::L) = x + (sum L);

(* mult : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: mult(L) evaluates to the product of the integers in L. *)
fun mult [ ] = 1
    | mult (x ::L) = x * (mult L)

(*****End*****)


val n = getInt();
val L = getIntList(n);
printInt(mult L);
