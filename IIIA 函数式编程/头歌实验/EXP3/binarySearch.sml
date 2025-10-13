fun printBool (a:bool) = 
    print(Bool.toString(a)^" "); 

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

fun split [ ]  = ([ ], [ ]) 
    | split [x] = ([ ], [x])
    | split (x::y::L) =
	let val (A, B) =split L
	in (x::A, y::B) 	
	end;

datatype tree = Empty | Br of tree * int * tree; 

fun trav(Br(t1,a,t2)) = trav(t1)@(a::trav(t2))
    |trav empty = [];

(*BEGIN*)
fun listToTree ([] : int list) : tree = Empty
  | listToTree (x::l) = 
    let
      val (leftList, rightList) = split l
      val leftTree = listToTree(leftList)
      val rightTree = listToTree(rightList)
    in
      Br(leftTree, x, rightTree)
    end;

fun binarySearch (Empty : tree, _ : int) : bool = false
  | binarySearch (Br(left, x, right), k) =
    case Int.compare(k, x) of
        EQUAL   => true
      | LESS    => binarySearch(left, k)
      | GREATER => binarySearch(right, k);
(*END*)

val L = getIntList(7);
val h= getInt();
printBool(binarySearch((listToTree L), h));
