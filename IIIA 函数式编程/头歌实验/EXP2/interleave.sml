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

 
(*** Begin ***)
(*** Begin ***)

(* interleave: int list * int list -> int list *)
fun interleave ([], []) = []
| interleave ([], L2) = L2
| interleave (L1, []) = L1
| interleave (x::xs, y::ys) = x :: y :: interleave(xs, ys)

(*****End*****)


(*****End*****)

val L = getIntList(2);
val R = getIntList(6);
val O = interleave (L, R);
printIntList(O); 
