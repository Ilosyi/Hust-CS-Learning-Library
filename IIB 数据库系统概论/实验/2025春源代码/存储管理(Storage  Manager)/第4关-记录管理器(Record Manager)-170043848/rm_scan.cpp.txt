#include "rm_scan.h"

#include "rm_file_handle.h"

/**
 * @brief 初始化file_handle和rid
 *
 * @param file_handle
 */
RmScan::RmScan(const RmFileHandle *file_handle) : file_handle_(file_handle) {
    // Todo:
    // 初始化file_handle和rid（指向第一个存放了记录的位置）
    int max_n = file_handle->file_hdr_.num_records_per_page;
    if(file_handle->file_hdr_.num_pages == 1) {
        rid_.slot_no = max_n;
        rid_.page_no = 0;
        return;
    }
    for(int i = 1; i < file_handle->file_hdr_.num_pages; ++i){
        rid_.page_no = i;
        RmPageHandle scanhead_page_handle = file_handle->fetch_page_handle(rid_.page_no);
        int slot_no = Bitmap::first_bit(true, scanhead_page_handle.bitmap, file_handle->file_hdr_.num_records_per_page);
        if(slot_no == file_handle->file_hdr_.num_records_per_page){
            continue;
        }else{
            rid_.page_no = i;
            rid_.slot_no = slot_no;
            break;
        }
    }
}

/**
 * @brief 找到文件中下一个存放了记录的位置
 */
void RmScan::next() {
    // Todo:
    // 找到文件中下一个存放了记录的非空闲位置，用rid_来指向这个位置
    int flag = 0;
    int init = rid_.page_no;
    for(int i = rid_.page_no; i < file_handle_->file_hdr_.num_pages; ++i){
        rid_.page_no = i;
        //printf("next我现在的page_no是%d\n",rid_.page_no);
        
        RmPageHandle scannext_page_handle = file_handle_->fetch_page_handle(rid_.page_no);
        int slot_no = -1;
        if(i == init){
            slot_no = Bitmap::next_bit(true, scannext_page_handle.bitmap, file_handle_->file_hdr_.num_records_per_page,rid_.slot_no);
        }
        else{
            slot_no = Bitmap::first_bit(true, scannext_page_handle.bitmap, file_handle_->file_hdr_.num_records_per_page);
        }
        if(slot_no != file_handle_->file_hdr_.num_records_per_page){
            rid_.page_no = i;
            rid_.slot_no = slot_no;
            flag = 1;
            //printf("我找到的next是page_no为%d,slot_no为%d\n",rid_.page_no,rid_.slot_no);
            break;
        }
    }
    if(flag == 0){
       
        rid_.page_no = file_handle_->file_hdr_.num_pages;
        rid_.slot_no = file_handle_->file_hdr_.num_records_per_page;
        //printf("找到头了，page_no是%d,slot_no是%d\n",rid_.page_no , rid_.slot_no);
    }
}

/**
 * @brief ​ 判断是否到达文件末尾
 */
bool RmScan::is_end() const {
    // Todo: 修改返回值
    return rid_.page_no == file_handle_->file_hdr_.num_pages && rid_.slot_no == file_handle_->file_hdr_.num_records_per_page;
}

/**
 * @brief RmScan内部存放的rid
 */
Rid RmScan::rid() const {
    // Todo: 修改返回值
    return rid_;
}