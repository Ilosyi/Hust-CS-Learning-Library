(* 
 * 函数签名: toInt : int -> int list -> int
 * 参数说明:
 *   - b : int           基数（进制），必须大于1
 *   - L : int list      数字列表，从低位到高位排列
 *                       如 [0,0,1,1] 表示二进制的1100
 * 返回值: 对应的十进制整数值
 *)

fun toInt b L =
    let
        fun helper [] _ = 0
          | helper (d::ds) power = d * power + helper ds (power * b)
    in
        helper L 1
    end;

(* 
 * 函数签名: toBase : int -> int -> int list
 * 参数说明:
 *   - b : int    目标基数（进制），必须大于1
 *   - n : int    待转换的十进制整数，必须≥0
 * 返回值: b进制表示的数字列表（从低位到高位）
 *)

fun toBase b n =
    if n = 0 then [0]
    else
        let
            fun helper 0 = []
              | helper m = (m mod b) :: helper (m div b)
        in
            helper n
        end;

(* 
 * 函数签名: convert : int * int -> int list -> int list
 * 参数说明:
 *   - (b1, b2) : int * int    源基数和目标基数的元组
 *   - L : int list            b1进制的数字列表
 * 返回值: b2进制的数字列表
 *)
fun convert (b1, b2) L = toBase b2 (toInt b1 L);

(* 测试用例 *)
(* 测试用例1：二进制与十进制的相互转换 *)
val test1_bin_to_dec = toInt 2 [0, 0, 1, 1];             (* 1100₂ = 12₁₀ *)
val test1_dec_to_bin = toBase 2 12;                      (* 12₁₀ = [0,0,1,1] *)
val test1_convert = convert (2, 10) [0, 0, 1, 1];       (* 1100₂ → 12₁₀ = [2,1] *)
(* 预期结果: test1_bin_to_dec = 12, test1_dec_to_bin = [0,0,1,1], test1_convert = [2,1] *)


(* 测试用例2：八进制转十六进制 *)
val test2_oct_to_dec = toInt 8 [5, 7, 2];                (* 275₈ = 189₁₀ *)
val test2_dec_to_hex = toBase 16 189;                    (* 189₁₀ = BD₁₆ = [13,11] *)
val test2_convert = convert (8, 16) [5, 7, 2];          (* 275₈ → BD₁₆ *)
(* 预期结果: test2_oct_to_dec = 189, test2_dec_to_hex = [13,11], test2_convert = [13,11] *)


(* 测试用例3：五进制转三进制 *)
val test3_base5 = toInt 5 [4, 3, 2];                     (* 234₅ = 69₁₀ *)
val test3_base3 = toBase 3 69;                           (* 69₁₀ = 2120₃ = [0,2,1,2] *)
val test3_convert = convert (5, 3) [4, 3, 2];           (* 234₅ → 2120₃ *)
(* 预期结果: test3_base5 = 69, test3_base3 = [0,2,1,2], test3_convert = [0,2,1,2] *)