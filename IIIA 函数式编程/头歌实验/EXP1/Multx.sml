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

fun getsubL ([], i) = ([],[])
   | getsubL (xs, 0) = ([],xs)
   | getsubL (x::xs, i) = 
   let 
    val (a,b) = getsubL(xs,i-1)
   in ((x::a),b)
 end;
    
    
(* 完成Begin和End间代码的编写  *)

(*****Begin*****)

(* multx : int list * int -> int      *)
(* REQUIRES: true      *)
(* ENSURES: multx(L, a) … (* FILL IN *)  *)

  fun multx ([ ], a) = a
  | multx (x :: L, a) = multx (L, x * a);

(* Multx: int list list * int -> int *)
(* REQUIRES: True            *)
(* ENSURES: Multx(R, a) evaluates to the product of integers in R times a*)

fun Multx( [ ], a) = a
  | Multx(r::R, a) = Multx(R, multx(r, a))
    
(*****end*****)


val L = getIntList(10);
val (L1,L2) = getsubL(L,3);
val (L3,L4) = getsubL(L2,2);
val L5 = L1::L3::L4::[];
printInt(Multx(L5,10));
