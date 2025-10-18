fun partition ([], pivot, less, greater) = (less, greater)
  | partition (x::xs, pivot, less, greater) =
      if x <= pivot then
          partition (xs, pivot, less @ [x], greater)
      else
          partition (xs, pivot, less, greater @ [x]);

fun quickSort [] = []
  | quickSort [x] = [x]
  | quickSort (pivot::xs) =
      let
          val (less, greater) = partition (xs, pivot, [], [])
          val sortedLess = quickSort less
          val sortedGreater = quickSort greater
      in
          sortedLess @ [pivot] @ sortedGreater
      end;

fun printIntList [] = print "\n"
  | printIntList (x::xs) =
      (print (Int.toString x ^ " "); printIntList xs);

print "test1: ";
printIntList (quickSort [1, 3, 8, 6, 11, 35 7]);

print "test2: ";
printIntList (quickSort [3, 1, 5, 1, 7, 11, 2, 8]);
