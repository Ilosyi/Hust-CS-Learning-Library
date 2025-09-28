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

(*  完成Begin和End间代码 *)    
(*****Begin*****)
fun reverse ([]) = []
  | reverse (x::xs) = (reverse xs) @ [x]

fun reverse2 L =
  let
    fun rev_helper ([], acc) = acc
      | rev_helper (x::xs, acc) = rev_helper (xs, x::acc)
  in
    rev_helper (L, [])
  end


(*****End*****)

val R = getIntList(5);
printIntList (reverse R);
printIntList (reverse2 R); 
