#include "rm_file_handle.h"

/**
 * @brief 由Rid得到指向RmRecord的指针
 *
 * @param rid 指定记录所在的位置
 * @return std::unique_ptr<RmRecord>
 */
std::unique_ptr<RmRecord> RmFileHandle::get_record(const Rid &rid, Context *context) const {
    // Todo:
    // 1. 获取指定记录所在的page handle
    // 2. 初始化一个指向RmRecord的指针（赋值其内部的data和size）
    RmPageHandle tmp_page_handle = fetch_page_handle(rid.page_no);
    int size_ = tmp_page_handle.file_hdr->record_size;
    char *data_ = tmp_page_handle.get_slot(rid.slot_no);
    //printf("在get_record里面为%s\n",data_);
    std::unique_ptr<RmRecord> record_ptr(new RmRecord(size_,data_));
    return record_ptr;
}

/**
 * @brief 在该记录文件（RmFileHandle）中插入一条记录
 *
 * @param buf 要插入的数据的地址
 * @return Rid 插入记录的位置
 */
Rid RmFileHandle::insert_record(char *buf, Context *context) {
    // Todo:
    // 1. 获取当前未满的page handle
    // 2. 在page handle中找到空闲slot位置
    // 3. 将buf复制到空闲slot位置
    // 4. 更新page_handle.page_hdr中的数据结构
    // 注意考虑插入一条记录后页面已满的情况，需要更新file_hdr_.first_free_page_no
    if(file_hdr_.first_free_page_no == -1){
        create_new_page_handle();
    }
        RmPageHandle insertpage_handle = fetch_page_handle(file_hdr_.first_free_page_no);
        Rid rid_;
        //rid_.slot_no = Bitmap::next_bit(true, page_handle.bitmap, file_handle_->file_hdr_.num_records_per_page,rid_.slot_no);
        int slot_no = Bitmap::first_bit(false, insertpage_handle.bitmap, file_hdr_.num_records_per_page);
        if(slot_no != file_hdr_.num_records_per_page){//也即找到了一个
            //复制数据进相应的slot
            memcpy(insertpage_handle.get_slot(slot_no),buf,file_hdr_.record_size);
            //改变bitmap
            Bitmap::set(insertpage_handle.bitmap,slot_no);
            //分配的record++
            insertpage_handle.page_hdr->num_records++;
            if(insertpage_handle.page_hdr->num_records >= file_hdr_.num_records_per_page){
                //说明插入了这个record之后这个页就满了，我们需要做相应的更新，需要更新满了之后的下一个页next_free_page_no和下一个，以及改变file_hdr_的first_free_page_no
                file_hdr_.first_free_page_no = insertpage_handle.page_hdr->next_free_page_no;
            }
        }
        buffer_pool_manager_->UnpinPage(insertpage_handle.page->GetPageId(), true);
        //printf("插入后立刻比较结果为%d\n",memcmp(get_record(Rid{insertpage_handle.page->GetPageId().page_no,slot_no},context)->data,buf,file_hdr_.record_size));
        //printf("插入后立即比较的结果是%d\n",memcmp(insertpage_handle.get_slot(slot_no),buf,file_hdr_.record_size));
        //printf("在insert里面buf为:%s\n",buf);
        //printf("在insert里插入后再取出来为%s\n",insertpage_handle.get_slot(slot_no));
        return Rid{insertpage_handle.page->GetPageId().page_no,slot_no};
}

/**
 * @brief 在该记录文件（RmFileHandle）中删除一条指定位置的记录
 *
 * @param rid 要删除的记录所在的指定位置
 */
void RmFileHandle::delete_record(const Rid &rid, Context *context) {
    // Todo:
    // 1. 获取指定记录所在的page handle
    // 2. 更新page_handle.page_hdr中的数据结构
    // 注意考虑删除一条记录后页面未满的情况，需要调用release_page_handle()
    RmPageHandle deletepage_handle = fetch_page_handle(rid.page_no);
    if(Bitmap::is_set(deletepage_handle.bitmap,rid.slot_no)){//如果被设置了，说明记录存在，那么处理它
        Bitmap::reset(deletepage_handle.bitmap,rid.slot_no);
        deletepage_handle.page_hdr->num_records--;
        if(deletepage_handle.page_hdr->num_records == (file_hdr_.num_records_per_page-1)){
            release_page_handle(deletepage_handle);
        }
        return ;
    }else{//如果这个记录本来就不存在，那么啥也不干
        return;
    }
}

/**
 * @brief 更新指定位置的记录
 *
 * @param rid 指定位置的记录
 * @param buf 新记录的数据的地址
 */
void RmFileHandle::update_record(const Rid &rid, char *buf, Context *context) {
    // Todo:
    // 1. 获取指定记录所在的page handle
    // 2. 更新记录
    RmPageHandle updatepage_handle = fetch_page_handle(rid.page_no);
    memcpy(updatepage_handle.get_slot(rid.slot_no),buf,file_hdr_.record_size);

}

/** -- 以下为辅助函数 -- */
/**
 * @brief 获取指定页面编号的page handle
 *
 * @param page_no 要获取的页面编号
 * @return RmPageHandle 返回给上层的page_handle
 * @note pin the page, remember to unpin it outside!
 */
RmPageHandle RmFileHandle::fetch_page_handle(int page_no) const {
    // Todo:
    // 使用缓冲池获取指定页面，并生成page_handle返回给上层
    // if page_no is invalid, throw PageNotExistError exception
    PageId page_id;
    page_id.fd = fd_;
    page_id.page_no = page_no;
    Page * fetch_page = buffer_pool_manager_->FetchPage(page_id);
    if(fetch_page == nullptr){
        const std::string temp("temp_table");
        throw PageNotExistError(temp,page_no);
        return RmPageHandle(&file_hdr_, nullptr);

    }
    return RmPageHandle(&file_hdr_,fetch_page);
}

/**
 * @brief 创建一个新的page handle
 *
 * @return RmPageHandle
 */
RmPageHandle RmFileHandle::create_new_page_handle() {
    // Todo:
    // 1.使用缓冲池来创建一个新page
    // 2.更新page handle中的相关信息
    // 3.更新file_hdr_
    PageId page_id;
    page_id.fd = fd_;
    Page * newpage = buffer_pool_manager_->NewPage(&page_id);//据说是移动了磁盘的一个新建的空page到bufferpool
    RmPageHandle newpage_handle = RmPageHandle(&file_hdr_,newpage);
    //如果这个页是新的。，那么更新page_hdr就好办了
    newpage_handle.page_hdr->next_free_page_no = file_hdr_.first_free_page_no;//新创建的插到列表前面
    newpage_handle.page_hdr->num_records = 0;//初始化为0
    file_hdr_.first_free_page_no = newpage->GetPageId().page_no;//移动第一个能用的指向这个页
    file_hdr_.num_pages ++;//多分了一个，那么就可以用
    return newpage_handle;
}

/**
 * @brief 创建或获取一个空闲的page handle
 *
 * @return RmPageHandle 返回生成的空闲page handle
 * @note pin the page, remember to unpin it outside!
 */
RmPageHandle RmFileHandle::create_page_handle() {
    // Todo:
    // 1. 判断file_hdr_中是否还有空闲页
    //     1.1 没有空闲页：使用缓冲池来创建一个新page；可直接调用create_new_page_handle()
    //     1.2 有空闲页：直接获取第一个空闲页
    // 2. 生成page handle并返回给上层
    if(file_hdr_.first_free_page_no == -1){
        return create_new_page_handle();
    }else{
        return fetch_page_handle(file_hdr_.first_free_page_no);
    }
}

/**
 * @brief 当page handle中的page从已满变成未满的时候调用
 *
 * @param page_handle
 * @note only used in delete_record()
 */
void RmFileHandle::release_page_handle(RmPageHandle &page_handle) {
    // Todo:
    // 当page从已满变成未满，考虑如何更新：
    // 1. page_handle.page_hdr->next_free_page_no
    // 2. file_hdr_.first_free_page_no
    page_handle.page_hdr->next_free_page_no = file_hdr_.first_free_page_no;
    file_hdr_.first_free_page_no = page_handle.page->GetPageId().page_no;
}

/**
 * @brief 用于事务的rollback操作
 *
 * @param rid record的插入位置
 * @param buf record的内容
 */
void RmFileHandle::insert_record(const Rid &rid, char *buf) {
    if (rid.page_no < file_hdr_.num_pages) {
        create_new_page_handle();
    }
    RmPageHandle pageHandle = fetch_page_handle(rid.page_no);
    Bitmap::set(pageHandle.bitmap, rid.slot_no);
    pageHandle.page_hdr->num_records++;
    if (pageHandle.page_hdr->num_records == file_hdr_.num_records_per_page) {
        file_hdr_.first_free_page_no = pageHandle.page_hdr->next_free_page_no;
    }

    char *slot = pageHandle.get_slot(rid.slot_no);
    memcpy(slot, buf, file_hdr_.record_size);

    buffer_pool_manager_->UnpinPage(pageHandle.page->GetPageId(), true);
}
