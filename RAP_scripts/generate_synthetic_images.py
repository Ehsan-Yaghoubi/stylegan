import os
import cv2
import RAP_utils
import random
import numpy as np
import rap_data_loading as rap


def get_head_area(img, mask, keypoints):
    assert img is not None and mask is not None

    head_point = keypoints[rap.kp_head]
    neck_point = keypoints[rap.kp_neck]
    left_elbow = keypoints[rap.kp_left_elbow]
    right_elbow = keypoints[rap.kp_right_elbow]

    # @todo: treat case when persons is viewed from the back and the left and right side are reversed
    if left_elbow[0] > right_elbow[0]:
        left_elbow, right_elbow = right_elbow, left_elbow

    # head bounding rect in format: [x, y, w, h]
    approx_head_brect = [left_elbow[0], 0,
                        right_elbow[0] - left_elbow[0], neck_point[1]]

    head_area_mask = mask[approx_head_brect[1]: approx_head_brect[1] + approx_head_brect[3],
                    approx_head_brect[0]: approx_head_brect[0] + approx_head_brect[2]]
    _, contours,  _ = cv2.findContours(head_area_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # no head contours found
    if len(contours) == 0:
        return None, None, None

    head_contour = contours[0]
    head_contour = RAP_utils.translate_points(points=head_contour, translate_factor=(approx_head_brect[0], approx_head_brect[1]))

    head_brect = cv2.boundingRect(head_contour)
    head_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(head_mask, [head_contour], -1, (255, 255, 255), -1)
    # cv2.destroyWindow('head contour')
    # cv2.imshow('head contour', head_mask)

    return head_brect, head_contour, head_mask


def remove_head_area(img, head_brect,  head_mask):
    kernel = np.ones((7, 7), np.uint8)

    head_mask_enlarged = cv2.dilate(head_mask, kernel, iterations=5)
    img_head_removed = cv2.inpaint(img, head_mask_enlarged, head_brect[2]*0.1 , cv2.INPAINT_TELEA)
    return img_head_removed


def replace_head_rect(img1, mask1, keypoints1,
                     img2, mask2, keypoints2):
    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        return None

    result_image = remove_head_area(img1, head_brect1, head_mask1)

    head_img2 = img2[head_brect2[1]: head_brect2[1]+head_brect2[3],
                    head_brect2[0]: head_brect2[0]+head_brect2[2]]
    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))

    result_image[head_brect1[1]:head_brect1[1]+head_brect1[3],
                    head_brect1[0]:head_brect1[0]+head_brect1[2]] = head_img2


    return result_image


def replace_head_area(img1, mask1, keypoints1,
                     img2, mask2, keypoints2):

    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        return None

    result_image = remove_head_area(img1, head_brect1, head_mask1)

    head_img2 = img2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]
    mask_img2 = mask2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]

    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.resize(mask_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.cvtColor(mask_img2, cv2.COLOR_GRAY2BGR)
    mask_img2 = mask_img2.astype(np.bool)

    np.copyto(result_image[head_brect1[1]: head_brect1[1] + head_brect1[3],
                head_brect1[0]: head_brect1[0] + head_brect1[2]], head_img2, where=mask_img2, casting='unsafe')

    return result_image


def generate_syntethic_images(rap_data, num_images_to_generate, viewpoint = 1, other_attrs = None):

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data=rap_data, attrib_index=rap.attr_OcclusionUp, attrib_value=0))
    target_images = set(rap.get_images_with_attib(rap_data=rap_data, attrib_index=rap.attr_viewpoint, attrib_value=viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)

    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = rap.get_images_with_attib(rap_data=rap_data, attrib_index=attr, attrib_value=other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)

    for _ in range(0, num_images_to_generate):
        cv2.destroyAllWindows()

        img_name1 = random.choice(target_images)
        img_name2 = random.choice(target_images)

        img_path1 = os.path.join(rap.rap_images_dir, img_name1)
        mask_path1 = os.path.join(rap.rap_masks_dir, img_name1)
        keypoints1 = rap_data[img_name1]['keypoints']
        img1 = cv2.imread(img_path1)
        mask1 = rap.load_crop_rap_mask(mask_path1)

        img_path2 = os.path.join(rap.rap_images_dir, img_name2)
        mask_path2 = os.path.join(rap.rap_masks_dir, img_name2)
        keypoints2 = rap_data[img_name2]['keypoints']
        img2 = cv2.imread(img_path2)
        mask2 = rap.load_crop_rap_mask(mask_path2)


        generated_replaced_area = replace_head_area(img1, mask1, keypoints1,
                                                    img2, mask2, keypoints2)
        generated_replaced_rect = replace_head_rect(img1, mask1, keypoints1,
                                                    img2, mask2, keypoints2)


        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)
        # if generated_replaced_area is not None:
        #     cv2.imshow('replaced head - by mask', generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     cv2.imshow('replaced head - by brect', generated_replaced_rect)
        # cv2.waitKey()
        # display
        img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        concat_images = [img1, img2_display]
        if generated_replaced_area is not None:
            concat_images.append(generated_replaced_area)
        if generated_replaced_rect is not None:
            concat_images.append(generated_replaced_rect)

        display_img = cv2.hconcat(concat_images)
        cv2.imshow('morphing', display_img)
        cv2.waitKey()
    return


if __name__ == '__main__':

    rap_dataset = rap.load_rap_dataset(rap.rap_attribute_annotations, rap.rap_keypoints_json)
    additional_attrs = {rap.attr_Female: 1}
    num_images_to_generate = 30
    generate_syntethic_images(rap_dataset, num_images_to_generate, other_attrs=additional_attrs)


